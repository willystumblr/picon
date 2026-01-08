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
from tqdm import tqdm

litellm.drop_params = True

class EvaluatorAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            role=kwargs.get('role', "evaluator"),
            system_message=kwargs.get('system_message', ""),
            model=kwargs.get('model', "gemini/gemini-2.5-flash"),
            port=kwargs.get('port', None)
        )
        current_dir = os.path.dirname(os.path.abspath(__file__)) 
        self.REPEAT_PROMPT = ("You will be given a single question and two corresponding answers. Determine whether the two answers are essentially the same in meaning. "
                              "If they are, output TRUE. If they are not, output FALSE. "
                              "Do not output any additional explanation or text.")
        self.abstain_prompt = open(f"{current_dir}/prompts/abstain.txt").read()
        
        self.affirmative_prompt = (
            "Your task is to evaluate whether the interviewee's response affirms the search results provided.\n"
            "An affirmative response should clearly confirm the fact presented in the question/search result.\n"
            "**Affirmative (True):**\n"
            "The response clearly confirms the fact.\n"
            "- *Includes:* \"Yes\", \"Correct\", \"That's me\".\n\n"
            "**Non-Affirmative (False):**\n"
            "The response does not confirm the fact, is vague, or indicates uncertainty.\n"
            "- *Includes:* \"I don't know\", \"Maybe\", \"Not sure\", irrelevant answers.\n\n"
            "Please strictly follow the guidelines above when making your judgment."
        )
        self.results_dict = {
            'internal': {
                'score': {
                    "harmonic_mean": 0.0,
                    "responsiveness_score": 0.0,
                    "consistency_score": 0.0         
                },
                'conflict': {
                    "count": 0,
                    "details": []
                },
                'plausible': {
                    "count": 0,
                    "details": []
                },
                'uncooperative': {
                    "count": 0,
                    "details": []
                },
            },
            'external': {
                'score': {
                    "harmonic_mean": 0.0,
                    "affirmativeness_score": 0.0,
                    "consistency_score": 0.0,
                },
                'supported': {
                    "count": 0,
                    "details": []
                },
                'rejected': {
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

    def __generate_abstain(self, qa_pair: Dict[str, str]):
        class InternalResponsive(BaseModel):
            abstain: bool = Field(..., description="Whether the response is abstaining from answering")
            rationale: str = Field(..., description="The rationale behind the verdict")
            abstain_type: Literal["none", "refusal", "lack_info", "asking_back", "unrelated"] | None = Field(None, description="Type of abstention if abstain is True")
            
        class ExternalAffirmative(BaseModel):
            is_affirmative: bool = Field(..., description="Whether the response affirms the search result")
            rationale: str = Field(..., description="The rationale behind the verdict")
        
        if qa_pair['flag']=='affirmative':
            response_format = ExternalAffirmative
        else:
            response_format = InternalResponsive
            
        completion_kwargs_1 = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": self.abstain_prompt} if qa_pair['flag']=="cooperative" else {"role": "system", "content": self.affirmative_prompt},
                {"role": "user", "content": "Question: " + qa_pair['question'] + "\n\nAnswer: " + qa_pair['response']}
            ],
            reasoning_effort="low",
            response_format=response_format
        )
        if self.model.startswith("hosted_vllm/"):
            assert self.port is not None, "Port must be specified for hosted_vllm models."    
            completion_kwargs_1['api_base'] = f"http://localhost:{self.port}/v1"
        response_1 = get_completion(**completion_kwargs_1)
        self._calculate_cost(response_1)
        parsed_response = response_format.model_validate_json(response_1.choices[0].message.content)
        return parsed_response
    
    def __generate_verdict(self, messages: List[Dict[str, Any]], abstain_response: BaseModel):
        class InternalVerdictResponseFormat(BaseModel):
            verdict: Literal['conflict', 'plausible']
            rationale: str = Field(..., description="The rationale behind the verdict")
            
        class ExternalVerdictResponseFormat(BaseModel):
            verdict: Literal['supported', 'rejected']
            rationale: str = Field(..., description="The rationale behind the verdict")
        
        class Verdict(BaseModel):
            verdict: str = Field(..., description="The verdict of the evaluation")
            is_cooperative: bool | None = Field(None, description="Whether the response is cooperative to the question")
            is_affirmative: bool | None = Field(None, description="Whether the response affirms the search result")
            rationale: str = Field(..., description="The rationale behind the verdict")
        
        if messages[-3]['role']=='tool':
            response_format = ExternalVerdictResponseFormat
            flag = "affirmative"
        else:
            response_format = InternalVerdictResponseFormat
            flag = "cooperative"
        
        if flag == "cooperative":  # responsive
            if not abstain_response.abstain:
                completion_kwargs_2 = dict(
                    model=self.model,
                    messages=messages + [
                        {
                            "role": "user",
                            "content": "Based on the conversation so far, determine whether the latest user response is in conflict with or plausible given the previous responses. "
                        }
                    ],
                    reasoning_effort="low",
                    response_format=response_format
                )
            else:
                verdict = Verdict(
                    verdict="uncooperative",
                    is_cooperative=False,
                    is_affirmative=None,
                    rationale=abstain_response.rationale
                )
                return verdict
        else:  # affirmative
            if abstain_response.is_affirmative:
                completion_kwargs_2 = dict(
                    model=self.model,
                    messages=messages + [
                        {
                            "role": "user",
                            "content": "Based on the conversation so far, determine whether the latest user response is supported by or rejected by the search results provided. "
                        }
                    ],
                    reasoning_effort="low",
                    response_format=response_format
                )
            else:
                verdict = Verdict(
                    verdict="non-affirmative",
                    is_affirmative=False,
                    is_cooperative=None,
                    rationale=abstain_response.rationale
                )
                return verdict
        if self.model.startswith("hosted_vllm/"):
            assert self.port is not None, "Port must be specified for hosted_vllm models."    
            completion_kwargs_2['api_base'] = f"http://localhost:{self.port}/v1"
        response_2 = get_completion(**completion_kwargs_2)
        self._calculate_cost(response_2)
        parsed_response_2 = response_format.model_validate_json(response_2.choices[0].message.content)
        
        verdict = Verdict(
            verdict=parsed_response_2.verdict,
            is_cooperative=not abstain_response.abstain if flag=="cooperative" else None,
            is_affirmative=abstain_response.is_affirmative if flag=="affirmative" else None,
            rationale=parsed_response_2.rationale
        )
        
        return verdict

    def __verdict_sanity_check(self, verdict: BaseModel) -> bool:
        """Check if the verdict object has valid fields"""
        if verdict.verdict in ['conflict', 'plausible', 'uncooperative']:
            if verdict.is_cooperative:
                return verdict.verdict in ['conflict', 'plausible']
            else:
                return verdict.verdict == 'uncooperative'
        elif verdict.verdict in ['supported', 'rejected', 'non-affirmative']:
            if verdict.is_affirmative:
                return verdict.verdict in ['supported', 'rejected']
            else:
                return verdict.verdict == 'non-affirmative'
        return False
    
    def _find_turn_idx(self, idx: int, history: List[Turn]) -> int:
        """find the turn index in self.history corresponding to the idx-th user response in self.evaluator_history"""
        assert self.memory[idx]['role'] == 'user', "The provided index does not correspond to a user message."
        user_response = self.memory[idx]['content']
        for i, turn in enumerate(history):
            if turn.type != 'repeat':
                for env_obs in turn.environment_observation:
                    if env_obs.observation_type == "interviewee_response":
                        if env_obs.response.content == user_response and env_obs.response.question == self.memory[idx - 1]['content']:
                            question = env_obs.response.question
                            return i, question, user_response

    def consistency_eval(self, history: List[Turn]):
        messages_lists = []
        user_indices = []
        
        for i in range(len(self.memory)):
            if self.memory[i]['role']=='user':
                messages_lists.append(self.memory[:i+1])
                user_indices.append(i)
        
        qa_pairs = [{"question": self.memory[i-1]['content'], "response": self.memory[i]['content'], "flag": "affirmative" if self.memory[i-2]['role']=='tool' else "cooperative"} for i in user_indices]
        
        with ThreadPoolExecutor(max_workers=16) as executor:
            abstain_results = list(tqdm(executor.map(self.__generate_abstain, qa_pairs), 
                                       total=len(messages_lists), 
                                       desc="Abstain evaluation"))
        
        # find the first user indices where not abstaining
        for idx, parsed_response in zip(user_indices, abstain_results):
            if (qa_pairs[0]['flag']=='cooperative' and not parsed_response.abstain) or (qa_pairs[0]['flag']=='affirmative' and parsed_response.is_affirmative):
                break
                
        if idx == user_indices[-1]: # all abstained
            return # nothing to evaluate
        elif idx > user_indices[0]:
            user_indices = user_indices[user_indices.index(idx)+1:] # dropping all prior abstained user messages, also droping the first non-abstained message since no prior context
            messages_lists = messages_lists[messages_lists.index(messages_lists[user_indices.index(idx)])+1:]
            abstain_results = abstain_results[abstain_results.index(parsed_response)+1:]
        else:
            user_indices.pop(0)  # remove the first user message (no prior context for consistency)
            messages_lists.pop(0)
            abstain_results.pop(0)
                
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(self.__generate_verdict, messages_lists, abstain_results), 
                               total=len(messages_lists), 
                               desc="Consistency evaluation"))
        
        for idx, verdict in zip(user_indices, results):
            assert self.__verdict_sanity_check(verdict), "Sanity check failed for verdict."
            if verdict.verdict in ['conflict', 'plausible', 'uncooperative']: # internal eval
                if verdict.is_cooperative:
                    turn_idx, question, user_response = self._find_turn_idx(idx, history)
                    if verdict.verdict == 'conflict':
                        self.results_dict['internal']['conflict']['count'] += 1
                        self.results_dict['internal']['conflict']['details'].append({
                            'turn_index': turn_idx,
                            'question': question,
                            'response': user_response,
                            'rationale': verdict.rationale
                        })
                    elif verdict.verdict == 'plausible':
                        self.results_dict['internal']['plausible']['count'] += 1
                        self.results_dict['internal']['plausible']['details'].append({
                            'turn_index': turn_idx,
                            'question': question,
                            'response': user_response,
                            'rationale': verdict.rationale
                        })
                else:
                    assert verdict.verdict == 'uncooperative', "If is_cooperative is False, verdict must be 'uncooperative'"
                    turn_idx, question, user_response = self._find_turn_idx(idx, history)
                    self.results_dict['internal']['uncooperative']['count'] += 1
                    self.results_dict['internal']['uncooperative']['details'].append({
                        'turn_index': turn_idx,
                        'question': question,
                        'response': user_response,
                        'rationale': verdict.rationale
                    })
            elif verdict.verdict in ['supported', 'rejected', 'non-affirmative']: # external eval
                if verdict.is_affirmative:
                    turn_idx, question, user_response = self._find_turn_idx(idx, history)
                    if verdict.verdict == 'supported':
                        self.results_dict['external']['supported']['count'] += 1
                        self.results_dict['external']['supported']['details'].append({
                            'turn_index': turn_idx,
                            'question': question,
                            'response': user_response,
                            'rationale': verdict.rationale
                        })
                    elif verdict.verdict == 'rejected':
                        self.results_dict['external']['rejected']['count'] += 1
                        self.results_dict['external']['rejected']['details'].append({
                            'turn_index': turn_idx,
                            'question': question,
                            'response': user_response,
                            'rationale': verdict.rationale
                        })
                else:
                    assert verdict.verdict == 'non-affirmative', f"If is_affirmative is False, verdict must be 'non-affirmative': {verdict.verdict}"
                    turn_idx, question, user_response = self._find_turn_idx(idx, history)
                    self.results_dict['external']['non-affirmative']['count'] += 1
                    self.results_dict['external']['non-affirmative']['details'].append({
                        'turn_index': turn_idx,
                        'question': question,
                        'response': user_response,
                        'rationale': verdict.rationale
                    })
            else:
                raise ValueError("Invalid verdict received from evaluator.")
        
        # Calculate scores
        # Internal score: 2 * plausible ratio * responsive ratio / (plausible ratio + responsive ratio)
        total_internal = self.results_dict['internal']['plausible']['count'] + self.results_dict['internal']['conflict']['count'] + self.results_dict['internal']['uncooperative']['count']
        plausible_ratio = (self.results_dict['internal']['plausible']['count'] / 
                          (self.results_dict['internal']['conflict']['count'] + self.results_dict['internal']['plausible']['count'])) if (self.results_dict['internal']['conflict']['count'] + self.results_dict['internal']['plausible']['count']) > 0 else 0.0
        responsive_ratio = ((self.results_dict['internal']['plausible']['count'] + self.results_dict['internal']['conflict']['count']) / total_internal) if total_internal > 0 else 0.0
        if plausible_ratio + responsive_ratio > 0:
            internal_score = 2 * plausible_ratio * responsive_ratio / (plausible_ratio + responsive_ratio)
        else:
            internal_score = 0.0
        self.results_dict['internal']['score']['harmonic_mean'] = internal_score
        self.results_dict['internal']['score']['responsiveness_score'] = responsive_ratio
        self.results_dict['internal']['score']['consistency_score'] = plausible_ratio
        # External score: 2 * supported ratio * affirmative ratio / (supported ratio + affirmative ratio)
        total_external = self.results_dict['external']['supported']['count'] + self.results_dict['external']['rejected']['count'] + self.results_dict['external']['non-affirmative']['count']
        supported_ratio = (self.results_dict['external']['supported']['count'] / 
                           (self.results_dict['external']['supported']['count'] + self.results_dict['external']['rejected']['count'])) if (self.results_dict['external']['supported']['count'] + self.results_dict['external']['rejected']['count']) > 0 else 0.0
        affirmative_ratio = ((self.results_dict['external']['supported']['count'] + self.results_dict['external']['rejected']['count']) / total_external) if total_external > 0 else 0.0
        if supported_ratio + affirmative_ratio > 0:
            external_score = 2 * supported_ratio * affirmative_ratio / (supported_ratio + affirmative_ratio)
        else:
            external_score = 0.0
        self.results_dict['external']['score']['harmonic_mean'] = external_score
        self.results_dict['external']['score']['affirmativeness_score'] = affirmative_ratio
        self.results_dict['external']['score']['consistency_score'] = supported_ratio
    
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
        
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(lambda kwargs: get_completion(**kwargs), completion_kwargs_list),
                               total=len(completion_kwargs_list),
                               desc="Intra-session evaluation"))
        
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
            
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(lambda kwargs: get_completion(**kwargs), completion_kwargs_list),
                               total=len(completion_kwargs_list),
                               desc="Inter-session evaluation"))
        
        for i, res in enumerate(results):
            self._calculate_cost(res)
            judge = res.choices[0].message.content.strip() if res and res.choices and res.choices[0].message and res.choices[0].message.content else None
            while judge not in ["TRUE", "FALSE"]:
                logging.warning(f"Unexpected response for inter-session repeat score: {judge}. Retrying...")
                res = get_completion(**completion_kwargs_list[i])
                self._calculate_cost(res)
                judge = res.choices[0].message.content.strip() if res and res.choices and res.choices[0].message and res.choices[0].message.content else None
            self.results_dict['stability']['inter_session']['details'].append({
                "question": all_get_to_knows[0][i].environment_observation[0].response.question,
                "responses": [turn.environment_observation[0].response.content for turn in [all_get_to_knows[sess_idx][i] for sess_idx in range(len(all_get_to_knows))]],
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
            for future in tqdm(futures, desc="Overall evaluation progress", total=len(futures)):
                future.result()  # wait for all to complete
        end_time = time.time()
        logging.info(f"Evaluation completed in {end_time - start_time:.2f} seconds.")
        
        return Action(
            agent=self.role,
            action_type="respond",
            content=self.results_dict
        )
        
        