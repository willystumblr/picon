from typing import List, Dict, Any, Literal
import re
import time
from pydantic import BaseModel, Field, ValidationError
import logging
import litellm
import os
import json
import time
from src.agents.base_agent import Agent
from src.utils import get_completion
from src.schemas import Action, Observation, Turn
from concurrent.futures import ThreadPoolExecutor

litellm.drop_params = True

class EvaluatorAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            role=kwargs.get('role', "evaluator"),
            system_message=kwargs.get('system_message', ""),
            model=kwargs.get('model', "gemini/gemini-2.5-flash"),
            port=kwargs.get('port', None)
        )
        self.REPEAT_PROMPT = ("You will be given a single question and two corresponding answers. Determine whether the two answers are essentially the same in meaning. "
                              "If they are, output TRUE. If they are not, output FALSE. "
                              "Do not output any additional explanation or text.")
        self.results_dict = {
            'internal': {
                'score': 0.0,
                'conflict': {
                    "count": 0,
                    "details": []
                },
                'plausible': {
                    "count": 0,
                    "details": []
                },
                'non-responsive': {
                    "count": 0,
                    "details": []
                },
            },
            'external': {
                'score': 0.0,
                'supported': {
                    "count": 0,
                    "details": []
                },
                'unsupported': {
                    "count": 0,
                    "details": []
                },
                'non-affirmative': {
                    "count": 0,
                    "details": []
                }
            },
            'stability': {
                'inter_session': {
                    'score': 0.0,
                    "details": []
                },
                'intra_session': {
                    'score': 0.0,
                    "details": []
                }
            }
        }

    def set_cutoff_date(self, cutoff_date: str) -> None:
        self.memory[0]['content'] = self.memory[0]['content'].format(cutoff_date=cutoff_date)

    def update_memory(self, **kwargs) -> None:
        if kwargs.get('index') is not None:
            index = kwargs.pop('index')
            assert 0 <= index <= len(self.memory), "Index out of range"
            # insert at specific index
            if index == len(self.memory):
                self.memory.append(kwargs)
            else:
                self.memory.insert(index, kwargs)
        else:
            self.memory.append(kwargs) # typically role and content

    def __generate_verdict(self, messages: List[Dict[str, Any]]):
        class InternalEvalResponse(BaseModel):
            is_responsive: bool = Field(..., description="Whether the response is responsive (engaged) to the question")
            verdict: Literal['conflict', 'plausible', 'non-responsive']
            rationale: str = Field(..., description="The rationale behind the verdict")

        class ExternalEvalResponse(BaseModel):
            is_affirmative: bool = Field(..., description="Whether the response affirms the search result")
            verdict: Literal['supported', 'unsupported', 'non-affirmative']
            rationale: str = Field(..., description="The rationale behind the verdict")
        
        if messages[-3]['role']=='tool':
            response_format = ExternalEvalResponse
        else:
            response_format = InternalEvalResponse

        completion_kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=0.7,
            response_format=response_format
        )
        if self.model.startswith("hosted_vllm/"):
            assert self.port is not None, "Port must be specified for hosted_vllm models."    
            completion_kwargs['api_base'] = f"http://localhost:{self.port}/v1"    
        verdict = get_completion(**completion_kwargs)
        self._calculate_cost(verdict)
        return verdict, response_format

    def _find_turn_idx(self, idx: int, history: List[Turn]) -> int:
        """find the turn index in self.history corresponding to the idx-th user response in self.evaluator_history"""
        assert self.memory[idx]['role'] == 'user', "The provided index does not correspond to a user message."
        user_response = self.memory[idx]['content']
        for i, turn in enumerate(history):
            if turn.type != 'repeat':
                for env_obs in turn.environment_observation:
                    if env_obs.observation_type == "interviewee_response":
                        if env_obs.response.content == user_response:
                            question = env_obs.response.question
                            return i, question, user_response

    def consistency_eval(self, history: List[Turn]):
        messages_lists = []
        user_indices = []
        
        for i in range(len(self.memory)):
            if self.memory[i]['role']=='user':
                messages_lists.append(self.memory[:i+1])
                user_indices.append(i)
                
        with ThreadPoolExecutor(max_workers=len(messages_lists)) as executor:
            results = list(executor.map(self.__generate_verdict, messages_lists))
        
        for idx, (verdict, response_format) in zip(user_indices, results):
            self._calculate_cost(verdict)
            parsed_verdict = response_format.model_validate_json(verdict.choices[0].message.content)
            if parsed_verdict.verdict in ['conflict', 'plausible', 'non-responsive']: # internal eval
                if parsed_verdict.is_responsive:
                    turn_idx, question, user_response = self._find_turn_idx(idx, history)
                    if parsed_verdict.verdict == 'conflict':
                        self.results_dict['internal']['conflict']['count'] += 1
                        self.results_dict['internal']['conflict']['details'].append({
                            'turn_index': turn_idx,
                            'question': question,
                            'response': user_response,
                            'rationale': parsed_verdict.rationale
                        })
                    elif parsed_verdict.verdict == 'plausible':
                        self.results_dict['internal']['plausible']['count'] += 1
                        self.results_dict['internal']['plausible']['details'].append({
                            'turn_index': turn_idx,
                            'question': question,
                            'response': user_response,
                            'rationale': parsed_verdict.rationale
                        })
                else:
                    assert parsed_verdict.verdict == 'non-responsive', "If is_responsive is False, verdict must be 'non-responsive'"
                    turn_idx, question, user_response = self._find_turn_idx(idx, history)
                    self.results_dict['internal']['non-responsive']['count'] += 1
                    self.results_dict['internal']['non-responsive']['details'].append({
                        'turn_index': turn_idx,
                        'question': question,
                        'response': user_response,
                        'rationale': parsed_verdict.rationale
                    })
            elif parsed_verdict.verdict in ['supported', 'unsupported', 'non-affirmative']: # external eval
                if parsed_verdict.is_affirmative:
                    turn_idx, question, user_response = self._find_turn_idx(idx, history)
                    if parsed_verdict.verdict == 'supported':
                        self.results_dict['external']['supported']['count'] += 1
                        self.results_dict['external']['supported']['details'].append({
                            'turn_index': turn_idx,
                            'question': question,
                            'response': user_response,
                            'rationale': parsed_verdict.rationale
                        })
                    elif parsed_verdict.verdict == 'unsupported':
                        self.results_dict['external']['unsupported']['count'] += 1
                        self.results_dict['external']['unsupported']['details'].append({
                            'turn_index': turn_idx,
                            'question': question,
                            'response': user_response,
                            'rationale': parsed_verdict.rationale
                        })
                else:
                    assert parsed_verdict.verdict == 'non-affirmative', "If is_affirmative is False, verdict must be 'non-affirmative'"
                    turn_idx, question, user_response = self._find_turn_idx(idx, history)
                    self.results_dict['external']['non-affirmative']['count'] += 1
                    self.results_dict['external']['non-affirmative']['details'].append({
                        'turn_index': turn_idx,
                        'question': question,
                        'response': user_response,
                        'rationale': parsed_verdict.rationale
                    })
            else:
                raise ValueError("Invalid verdict received from evaluator.")
        
        # Calculate scores
        # Internal score: 2 * plausible ratio * responsive ratio / (plausible ratio + responsive ratio)
        total_internal = self.results_dict['internal']['plausible']['count'] + self.results_dict['internal']['conflict']['count'] + self.results_dict['internal']['non-responsive']['count']
        plausible_ratio = (self.results_dict['internal']['plausible']['count'] / 
                          (self.results_dict['internal']['conflict']['count'] + self.results_dict['internal']['plausible']['count'])) if (self.results_dict['internal']['conflict']['count'] + self.results_dict['internal']['plausible']['count']) > 0 else 0.0
        responsive_ratio = ((self.results_dict['internal']['plausible']['count'] + self.results_dict['internal']['conflict']['count']) / total_internal) if total_internal > 0 else 0.0
        if plausible_ratio + responsive_ratio > 0:
            internal_score = 2 * plausible_ratio * responsive_ratio / (plausible_ratio + responsive_ratio)
        else:
            internal_score = 0.0
        self.results_dict['internal']['score'] = internal_score
        # External score: 2 * supported ratio * affirmative ratio / (supported ratio + affirmative ratio)
        total_external = self.results_dict['external']['supported']['count'] + self.results_dict['external']['unsupported']['count'] + self.results_dict['external']['non-affirmative']['count']
        supported_ratio = (self.results_dict['external']['supported']['count'] / 
                           (self.results_dict['external']['supported']['count'] + self.results_dict['external']['unsupported']['count'])) if (self.results_dict['external']['supported']['count'] + self.results_dict['external']['unsupported']['count']) > 0 else 0.0
        affirmative_ratio = ((self.results_dict['external']['supported']['count'] + self.results_dict['external']['unsupported']['count']) / total_external) if total_external > 0 else 0.0
        if supported_ratio + affirmative_ratio > 0:
            external_score = 2 * supported_ratio * affirmative_ratio / (supported_ratio + affirmative_ratio)
        else:
            external_score = 0.0
        self.results_dict['external']['score'] = external_score
    
    def intra_session_eval(self, history: List[Turn]):
        completion_kwargs_list = []
        get_to_knows = [turn for turn in history if turn.type == 'get_to_know']
        repeats = [turn for turn in history if turn.type == 'repeat']
        for original, repeat in zip(get_to_knows, repeats):
            assert original.environment_observation[0].response.question in repeat.environment_observation[0].response.question, "Mismatch in questions between original and repeat."
            
            completion_kwargs = dict(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.REPEAT_PROMPT},
                    {"role": "user", "content": f"Question: {original.environment_observation[0].response.question}\n\nResponse 1: {original.environment_observation[0].response.content}\nResponse 2: {repeat.environment_observation[0].response.content}"}
                ],
                reasoning_effort="low",
            )
            if self.model.startswith("hosted_vllm/"):
                assert self.port is not None, "Port must be specified for hosted_vllm models."    
                completion_kwargs['api_base'] = f"http://localhost:{self.port}/v1"
            completion_kwargs_list.append(completion_kwargs)
        
        with ThreadPoolExecutor(max_workers=len(completion_kwargs_list)) as executor:
            results = list(executor.map(lambda kwargs: get_completion(**kwargs), completion_kwargs_list))
        
        for i, res in enumerate(results):
            self._calculate_cost(res) if not self.model.startswith("hosted_vllm/") else 0.0
            judge = res.choices[0].message.content.strip() if res and res.choices and res.choices[0].message and res.choices[0].message.content else None
            while judge not in ["TRUE", "FALSE"]:
                logging.warning(f"Unexpected response for repeat score: {judge}. Retrying...")
                res = get_completion(**completion_kwargs_list[i])
                self._calculate_cost(res)
                judge = res.choices[0].message.content.strip() if res and res.choices and res.choices[0].message and res.choices[0].message.content else None
            self.results_dict['stability']['intra_session']['details'].append({
                "question": f"Just to clarify, {get_to_knows[i].environment_observation[0].response.question}",
                "original_response": get_to_knows[i].environment_observation[0].response.content,
                "repeated_response": repeats[i].environment_observation[0].response.content,
                "is_aligned": judge
            })
            self.results_dict['stability']['intra_session']['score'] += (judge=='TRUE')
        self.results_dict['stability']['intra_session']['score'] = round(self.results_dict['stability']['intra_session']['score'] / len(get_to_knows), 4)
    
    def inter_session_eval(self, histories: List[List[Turn]]):
        # collect all get_to_know turns across sessions
        all_get_to_knows = []
        for history in histories:
            get_to_knows = [turn for turn in history if turn.type == 'get_to_know']
            all_get_to_knows.append(get_to_knows)
        # assert all sessions have the same number of get_to_know turns
        num_turns = len(all_get_to_knows[0])
        for get_to_knows in all_get_to_knows:
            assert len(get_to_knows) == num_turns, "Mismatch in number of get_to_know turns across sessions."
        
        # create a string with all responses for each turn
        completion_kwargs_list = []
        for turn_idx in range(num_turns):
            responses = []
            question = all_get_to_knows[0][turn_idx].environment_observation[0].response.question
            for session_idx, get_to_knows in enumerate(all_get_to_knows):
                response_content = get_to_knows[turn_idx].environment_observation[0].response.content
                responses.append(f"Response from session {session_idx+1}: {response_content}")
            combined_responses = "\n\n".join(responses)
            completion_kwargs = dict(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.REPEAT_PROMPT},
                    {"role": "user", "content": f"Question: {question}\n\n{combined_responses}"}
                ],
                reasoning_effort="low",
            )
            if self.model.startswith("hosted_vllm/"):
                assert self.port is not None, "Port must be specified for hosted_vllm models."    
                completion_kwargs['api_base'] = f"http://localhost:{self.port}/v1"
            completion_kwargs_list.append(completion_kwargs)
            
        with ThreadPoolExecutor(max_workers=len(completion_kwargs_list)) as executor:
            results = list(executor.map(lambda kwargs: get_completion(**kwargs), completion_kwargs_list))
        
        for i, res in enumerate(results):
            self._calculate_cost(res)
            judge = res.choices[0].message.content.strip() if res and res.choices and res.choices[0].message and res.choices[0].message.content else None
            while judge not in ["TRUE", "FALSE"]:
                logging.warning(f"Unexpected response for inter-session repeat score: {judge}. Retrying...")
                res = get_completion(**completion_kwargs_list[i])
                self._calculate_cost(res)
                judge = res.choices[0].message.content.strip() if res and res.choices and res.choices[0].message and res.choices[0].message.content else None
            self.results_dict['stability']['inter_session']['details'].append({
                "question": question,
                "responses": [get_to_knows[turn_idx].environment_observation[0].response.content for get_to_knows in all_get_to_knows],
                "is_aligned": judge
            })
            self.results_dict['stability']['inter_session']['score'] += (judge=='TRUE')
        self.results_dict['stability']['inter_session']['score'] = round(self.results_dict['stability']['inter_session']['score'] / num_turns, 4)
    def act(self, histories: List[List[Turn]]) -> Action:
        main_history = histories[0]
        
        # parallel execution of consistency, intra-session, inter-session evals
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            futures.append(executor.submit(self.consistency_eval, main_history))
            futures.append(executor.submit(self.intra_session_eval, main_history))
            if len(histories) > 1:
                futures.append(executor.submit(self.inter_session_eval, histories))
            for future in futures:
                future.result()  # wait for all to complete
        end_time = time.time()
        logging.info(f"Evaluation completed in {end_time - start_time:.2f} seconds.")
        
        return Action(
            agent=self.role,
            action_type="respond",
            content=self.results_dict
        )
        
        