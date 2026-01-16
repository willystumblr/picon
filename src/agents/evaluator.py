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
from src.schemas import Action, Observation, Turn, ToolOutput
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

litellm.drop_params = True


class EvaluatorAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            role=kwargs.get('role', "evaluator"),
            system_message=kwargs.get('system_message', ""),
            model=kwargs.get('model', "gemini/gemini-2.5-flash"),
            port=kwargs.get('port', None),
            host=kwargs.get('host', 'localhost')
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
        self.affirmed_search_results = []  # Store affirmed search results for external consistency check
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
                'irrelevant': {
                    "count": 0,
                    "details": []
                },
                'plausible': {
                    "count": 0,
                    "details": []
                },
                'conflict': {
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

    def __generate_uncooperative_check(self, qa_pair: Dict[str, Any]):
        """Check if the answer is uncooperative (Step 1)"""
        class UncooperativeResponse(BaseModel):
            is_uncooperative: bool = Field(..., description="Whether the response is uncooperative (abstaining from answering)")
            rationale: str = Field(..., description="The rationale behind the verdict")
            uncooperative_type: Literal["none", "refusal", "lack_info", "asking_back", "unrelated"] | None = Field(None, description="Type of uncooperative response")
        
        user_content = "Question: " + qa_pair['question'] + "\n\nAnswer: " + qa_pair['response']
        
        if qa_pair.get('log_prompt', False):
            logging.info(f"[Uncooperative Check] System Prompt:\n{self.abstain_prompt}")
            logging.info(f"[Uncooperative Check] User Prompt:\n{user_content}")
            
        completion_kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": self.abstain_prompt},
                {"role": "user", "content": user_content}
            ],
            reasoning_effort="low",
            response_format=UncooperativeResponse
        )
        if self.model.startswith("hosted_vllm/"):
            assert self.port is not None, "Port must be specified for hosted_vllm models."    
            completion_kwargs['api_base'] = f"http://{self.host}:{self.port}/v1"
        response = get_completion(**completion_kwargs)
        self._calculate_cost(response)
        parsed_response = UncooperativeResponse.model_validate_json(response.choices[0].message.content)
        return parsed_response
    
    def __generate_affirmative_check(self, qa_pair: Dict[str, Any]):
        """Check if the answer affirms the search result (Step 2 - only for confirmation questions)"""
        class AffirmativeResponse(BaseModel):
            is_affirmative: bool = Field(..., description="Whether the response affirms the search result")
            rationale: str = Field(..., description="The rationale behind the verdict")
        
        user_content = "Question: " + qa_pair['question'] + "\n\nAnswer: " + qa_pair['response']
        
        if qa_pair.get('log_prompt', False):
            logging.info(f"[Affirmative Check] System Prompt:\n{self.affirmative_prompt}")
            logging.info(f"[Affirmative Check] User Prompt:\n{user_content}")
        
        completion_kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": self.affirmative_prompt},
                {"role": "user", "content": user_content}
            ],
            reasoning_effort="low",
            response_format=AffirmativeResponse
        )
        if self.model.startswith("hosted_vllm/"):
            assert self.port is not None, "Port must be specified for hosted_vllm models."    
            completion_kwargs['api_base'] = f"http://{self.host}:{self.port}/v1"
        response = get_completion(**completion_kwargs)
        self._calculate_cost(response)
        parsed_response = AffirmativeResponse.model_validate_json(response.choices[0].message.content)
        return parsed_response
    
    def __generate_internal_consistency_verdict(self, qa_history: str, current_question: str, current_response: str, log_prompt: bool = False):
        """Generate internal consistency verdict (conflict/plausible) using previous Q&A without tool outputs"""
        class InternalVerdictResponseFormat(BaseModel):
            verdict: Literal['conflict', 'plausible']
            rationale: str = Field(..., description="The rationale behind the verdict")
        
        current_qa_str = f"\n\nCurrent Question: {current_question}\nCurrent Response: {current_response}"
        user_content = qa_history + current_qa_str + "\n\nBased on the previous Q&A above (without any search results), determine whether the Current Response is in conflict with or plausible given the previous responses."
        
        if log_prompt:
            logging.info(f"[Internal Consistency Check] System Prompt:\n{self.memory[0]['content']}")
            logging.info(f"[Internal Consistency Check] User Prompt:\n{user_content}")
        
        completion_kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": self.memory[0]['content']},
                {"role": "user", "content": user_content}
            ],
            reasoning_effort="low",
            response_format=InternalVerdictResponseFormat
        )
        
        if self.model.startswith("hosted_vllm/"):
            assert self.port is not None, "Port must be specified for hosted_vllm models."    
            completion_kwargs['api_base'] = f"http://{self.host}:{self.port}/v1"
        
        response = get_completion(**completion_kwargs)
        self._calculate_cost(response)
        parsed_response = InternalVerdictResponseFormat.model_validate_json(response.choices[0].message.content)
        
        return parsed_response
    
    def __generate_external_consistency_verdict(self, affirmed_search_results: List[str], current_question: str, current_response: str, log_prompt: bool = False):
        """Generate external consistency verdict (irrelevant/plausible/conflict) using affirmed search results"""
        class ExternalVerdictResponseFormat(BaseModel):
            verdict: Literal['irrelevant', 'plausible', 'conflict']
            rationale: str = Field(..., description="The rationale behind the verdict")
        
        search_results_str = "Affirmed Search Results:\n" + "\n\n".join(affirmed_search_results)
        current_qa_str = f"\n\nCurrent Question: {current_question}\nCurrent Response: {current_response}"
        user_content = search_results_str + current_qa_str + """\n\nBased on the affirmed search results above, classify the Current Response into one of three categories:

1. **irrelevant**: The Q&A is completely unrelated to all affirmed search results. There is no overlapping information at all.
2. **conflict**: The answer contradicts or is inconsistent with at least one affirmed search result.
3. **plausible**: The answer is related to some affirmed search results but does not conflict with them.

Determine which category best fits the Current Response."""
        
        if log_prompt:
            logging.info(f"[External Consistency Check] System Prompt:\n{self.memory[0]['content']}")
            logging.info(f"[External Consistency Check] User Prompt:\n{user_content}")
            logging.info(f"[External Consistency Check] Number of affirmed search results: {len(affirmed_search_results)}")
        
        completion_kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": self.memory[0]['content']},
                {"role": "user", "content": user_content}
            ],
            reasoning_effort="low",
            response_format=ExternalVerdictResponseFormat
        )
        
        if self.model.startswith("hosted_vllm/"):
            assert self.port is not None, "Port must be specified for hosted_vllm models."    
            completion_kwargs['api_base'] = f"http://{self.host}:{self.port}/v1"
        
        response = get_completion(**completion_kwargs)
        self._calculate_cost(response)
        parsed_response = ExternalVerdictResponseFormat.model_validate_json(response.choices[0].message.content)
        
        return parsed_response
    
    def _extract_tool_output_str(self, tool_output: 'ToolOutput') -> str:
        """Extract tool output string including claims and output."""
        claims_str = ""
        if tool_output.arguments and 'claims' in tool_output.arguments:
            claims = tool_output.arguments['claims']
            if claims:
                claims_str = "Claims: " + "; ".join(claims) + "\n"
        output_str = f"Output: {tool_output.output}"
        return claims_str + output_str

    def consistency_eval(self, history: List[Turn]):
        """
        Evaluate consistency using history directly instead of evaluator's memory.
        
        For every answer from user:
        1. Check if the answer is uncooperative
        2. If confirmation question, check if affirmative and save affirmed search results
        3. Internal check: previous Q&A (without tool outputs) + current Q&A
        4. External check: affirmed search results + current Q&A (skip if no affirmed results)
        """
        # Filter out repeat turns
        non_repeat_turns = [turn for turn in history if turn.type != 'repeat']
        
        if not non_repeat_turns:
            return  # nothing to evaluate
        
        # Reset affirmed search results
        self.affirmed_search_results = []
        
        # Build evaluation items from history
        eval_items = []
        qa_history = "Previous Q&A:\n\n"  # For internal check (Q&A only, no tool outputs)
        
        for turn_idx, turn in enumerate(non_repeat_turns):
            env_obs = turn.environment_observation
            if not env_obs:
                continue
            
            # First observation is always an interviewee response
            first_obs = env_obs[0]
            if first_obs.observation_type == "interviewee_response" and first_obs.response:
                question = first_obs.response.question
                response = first_obs.response.content
                
                # Add to eval items (regular question)
                eval_items.append({
                    'turn_idx': turn_idx,
                    'question': question,
                    'response': response,
                    'is_confirmation': False,
                    'tool_output_str': None,
                    'qa_history': qa_history  # Q&A history up to this point (for internal check)
                })
                
                # Update Q&A history (without tool outputs)
                qa_history += f"Q: {question}\nA: {response}\n\n"
            
            # Collect tool outputs from this turn
            tool_outputs = []
            for obs in env_obs:
                if obs.observation_type == "tool_output" and obs.tool_output:
                    tool_outputs.extend(obs.tool_output)
            
            # Process subsequent interviewee_responses (confirmation questions)
            confirmation_responses = []
            for obs in env_obs[1:]:  # skip first observation
                if obs.observation_type == "interviewee_response" and obs.response:
                    confirmation_responses.append(obs)
            
            # Match tool outputs to confirmation responses by index
            for i, tool_output in enumerate(tool_outputs):
                tool_output_str = self._extract_tool_output_str(tool_output)
                
                # Check if there's a matching confirmation response
                if i < len(confirmation_responses):
                    conf_obs = confirmation_responses[i]
                    conf_question = conf_obs.response.question
                    conf_response = conf_obs.response.content
                    
                    # Add to eval items (confirmation question)
                    eval_items.append({
                        'turn_idx': turn_idx,
                        'question': conf_question,
                        'response': conf_response,
                        'is_confirmation': True,
                        'tool_output_str': tool_output_str,
                        'qa_history': qa_history  # Q&A history up to this point (for internal check)
                    })
                    
                    # Update Q&A history with confirmation Q&A (without tool output)
                    qa_history += f"Q: {conf_question}\nA: {conf_response}\n\n"
        
        if not eval_items:
            return  # nothing to evaluate
        
        # Step 1: Check uncooperative for ALL items
        qa_pairs = [{
            "question": item['question'], 
            "response": item['response'], 
            "log_prompt": (i < 3)
        } for i, item in enumerate(eval_items)]
        
        with ThreadPoolExecutor(max_workers=16) as executor:
            uncooperative_results = list(tqdm(executor.map(self.__generate_uncooperative_check, qa_pairs), 
                                              total=len(qa_pairs), 
                                              desc="Uncooperative check"))
        
        # Step 2: Check affirmative for confirmation questions only
        confirmation_items_indices = [i for i, item in enumerate(eval_items) if item['is_confirmation']]
        confirmation_qa_pairs = [{
            "question": eval_items[i]['question'], 
            "response": eval_items[i]['response'], 
            "log_prompt": (j < 3)
        } for j, i in enumerate(confirmation_items_indices)]
        
        affirmative_results = [None] * len(eval_items)  # Initialize with None for non-confirmation items
        if confirmation_qa_pairs:
            with ThreadPoolExecutor(max_workers=16) as executor:
                confirmation_affirmative_results = list(tqdm(executor.map(self.__generate_affirmative_check, confirmation_qa_pairs), 
                                                              total=len(confirmation_qa_pairs), 
                                                              desc="Affirmative check"))
            for idx, result in zip(confirmation_items_indices, confirmation_affirmative_results):
                affirmative_results[idx] = result
        
        # Store uncooperative results and compute ratios
        cooperative_count = 0
        total_count = len(eval_items)
        
        for item, uncoop_result in zip(eval_items, uncooperative_results):
            if uncoop_result.is_uncooperative:
                self.results_dict['internal']['uncooperative']['count'] += 1
                self.results_dict['internal']['uncooperative']['details'].append({
                    'turn_index': item['turn_idx'],
                    'question': item['question'],
                    'response': item['response'],
                    'rationale': uncoop_result.rationale,
                    'uncooperative_type': uncoop_result.uncooperative_type
                })
            else:
                cooperative_count += 1
        
        responsive_ratio = cooperative_count / total_count if total_count > 0 else 0.0
        
        # Store affirmative/non-affirmative results and save affirmed search results
        affirmative_count = 0
        total_confirmation_count = len(confirmation_items_indices)
        
        for idx in confirmation_items_indices:
            item = eval_items[idx]
            aff_result = affirmative_results[idx]
            if aff_result.is_affirmative:
                affirmative_count += 1
                # Save affirmed search result
                self.affirmed_search_results.append(item['tool_output_str'])
            else:
                self.results_dict['external']['non-affirmative']['count'] += 1
                self.results_dict['external']['non-affirmative']['details'].append({
                    'turn_index': item['turn_idx'],
                    'question': item['question'],
                    'response': item['response'],
                    'rationale': aff_result.rationale
                })
        
        affirmative_ratio = affirmative_count / total_confirmation_count if total_confirmation_count > 0 else 0.0
        
        # Step 3: Internal consistency check (need at least 2 items for previous context)
        # Step 4: External consistency check (all Q&A with all affirmed results)
        
        # Internal check: skip first item as it has no prior context
        if len(eval_items) >= 2:
            consistency_eval_items = eval_items[1:]
            consistency_uncoop_results = uncooperative_results[1:]
            
            # Internal consistency check for all items (from 2nd item)
            def generate_internal_wrapper(args):
                item, idx = args
                return self.__generate_internal_consistency_verdict(item['qa_history'], item['question'], item['response'], log_prompt=(idx < 3))
            
            with ThreadPoolExecutor(max_workers=16) as executor:
                internal_results = list(tqdm(executor.map(generate_internal_wrapper, [(item, idx) for idx, item in enumerate(consistency_eval_items)]), 
                                            total=len(consistency_eval_items), 
                                            desc="Internal consistency check"))
            
            # Process internal consistency results
            for item, uncoop_result, internal_verdict in zip(consistency_eval_items, consistency_uncoop_results, internal_results):
                turn_idx = item['turn_idx']
                question = item['question']
                user_response = item['response']
                
                if internal_verdict.verdict == 'conflict':
                    self.results_dict['internal']['conflict']['count'] += 1
                    self.results_dict['internal']['conflict']['details'].append({
                        'turn_index': turn_idx,
                        'question': question,
                        'response': user_response,
                        'rationale': internal_verdict.rationale,
                        'is_cooperative': not uncoop_result.is_uncooperative
                    })
                elif internal_verdict.verdict == 'plausible':
                    self.results_dict['internal']['plausible']['count'] += 1
                    self.results_dict['internal']['plausible']['details'].append({
                        'turn_index': turn_idx,
                        'question': question,
                        'response': user_response,
                        'rationale': internal_verdict.rationale,
                        'is_cooperative': not uncoop_result.is_uncooperative
                    })
        
        # External check: all Q&A (excluding confirmation questions) with ALL affirmed results
        if self.affirmed_search_results:
            all_affirmed = self.affirmed_search_results  # Use all affirmed results from entire conversation
            
            # Filter out confirmation questions for external check
            non_confirmation_items = [(item, idx) for idx, item in enumerate(eval_items) if not item['is_confirmation']]
            
            def generate_external_wrapper(args):
                item, idx = args
                return self.__generate_external_consistency_verdict(all_affirmed, item['question'], item['response'], log_prompt=(idx < 3))
            
            with ThreadPoolExecutor(max_workers=16) as executor:
                external_results = list(tqdm(executor.map(generate_external_wrapper, non_confirmation_items), 
                                            total=len(non_confirmation_items), 
                                            desc="External consistency check"))
            
            # Process external consistency results
            for (item, _), external_verdict in zip(non_confirmation_items, external_results):
                turn_idx = item['turn_idx']
                question = item['question']
                user_response = item['response']
                
                if external_verdict.verdict == 'irrelevant':
                    self.results_dict['external']['irrelevant']['count'] += 1
                    self.results_dict['external']['irrelevant']['details'].append({
                        'turn_index': turn_idx,
                        'question': question,
                        'response': user_response,
                        'rationale': external_verdict.rationale
                    })
                elif external_verdict.verdict == 'conflict':
                    self.results_dict['external']['conflict']['count'] += 1
                    self.results_dict['external']['conflict']['details'].append({
                        'turn_index': turn_idx,
                        'question': question,
                        'response': user_response,
                        'rationale': external_verdict.rationale
                    })
                elif external_verdict.verdict == 'plausible':
                    self.results_dict['external']['plausible']['count'] += 1
                    self.results_dict['external']['plausible']['details'].append({
                        'turn_index': turn_idx,
                        'question': question,
                        'response': user_response,
                        'rationale': external_verdict.rationale
                    })
        
        # Calculate final scores
        internal_plausible_ratio = (self.results_dict['internal']['plausible']['count'] / 
                                   (self.results_dict['internal']['conflict']['count'] + self.results_dict['internal']['plausible']['count'])) if (self.results_dict['internal']['conflict']['count'] + self.results_dict['internal']['plausible']['count']) > 0 else 0.0
        
        external_plausible_ratio = (self.results_dict['external']['plausible']['count'] / 
                                    (self.results_dict['external']['plausible']['count'] + self.results_dict['external']['conflict']['count'])) if (self.results_dict['external']['plausible']['count'] + self.results_dict['external']['conflict']['count']) > 0 else 0.0
        
        self._calculate_final_scores(responsive_ratio, affirmative_ratio, internal_plausible_ratio, external_plausible_ratio)
        
        # Save affirmed search results to final output
        self.results_dict['confirmed_search_results'] = self.affirmed_search_results
    
    def _calculate_final_scores(self, responsive_ratio: float, affirmative_ratio: float, internal_plausible_ratio: float, external_plausible_ratio: float):
        """Calculate and store final scores"""
        # Internal score: harmonic mean of plausible ratio and responsive ratio
        if internal_plausible_ratio + responsive_ratio > 0:
            internal_score = 2 * internal_plausible_ratio * responsive_ratio / (internal_plausible_ratio + responsive_ratio)
        else:
            internal_score = 0.0
        self.results_dict['internal']['score']['harmonic_mean'] = internal_score
        self.results_dict['internal']['score']['responsiveness_score'] = responsive_ratio
        self.results_dict['internal']['score']['consistency_score'] = internal_plausible_ratio
        
        # External score: harmonic mean of plausible ratio and affirmative ratio
        if external_plausible_ratio + affirmative_ratio > 0:
            external_score = 2 * external_plausible_ratio * affirmative_ratio / (external_plausible_ratio + affirmative_ratio)
        else:
            external_score = 0.0
        self.results_dict['external']['score']['harmonic_mean'] = external_score
        self.results_dict['external']['score']['affirmativeness_score'] = affirmative_ratio
        self.results_dict['external']['score']['consistency_score'] = external_plausible_ratio
    
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
                completion_kwargs['api_base'] = f"http://{self.host}:{self.port}/v1"
            completion_kwargs_list.append(completion_kwargs)
        
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(lambda kwargs: get_completion(**kwargs), completion_kwargs_list),
                               total=len(completion_kwargs_list),
                               desc="Intra-session evaluation"))
        
        for i, res in enumerate(results):
            self._calculate_cost(res) if not self.model.startswith("hosted_vllm/") else 0.0
            judge = res.choices[0].message.content.strip() if res and res.choices and res.choices[0].message and res.choices[0].message.content else None
            if judge is None:
                breakpoint()
            if '</think>' in judge:
                judge = judge.split('</think>')[-1].strip()
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
                completion_kwargs['api_base'] = f"http://{self.host}:{self.port}/v1"
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