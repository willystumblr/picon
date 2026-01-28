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
        self.external_eval_prompt = open(f"{current_dir}/prompts/evaluator_prompt.txt").read()
        
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
                'total_claims': 0,
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
                },
                'nei': {
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
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = get_completion(**completion_kwargs)
                self._calculate_cost(response)
                parsed_response = UncooperativeResponse.model_validate_json(response.choices[0].message.content)
                return parsed_response
            except ValidationError as e:
                if attempt == max_retries - 1:
                    logging.error(f"Pydantic validation failed after {max_retries} attempts: {e}")
                    raise
                logging.warning(f"Pydantic validation failed for UncooperativeResponse, retrying... ({attempt + 1}/{max_retries})")
                continue
    
    def __generate_external_turn_verdict(self, turn_data: Dict[str, Any], log_prompt: bool = False):
        """
        Generate external consistency verdict for a single turn with web search results.
        Evaluates each CLAIM individually and returns verdicts for all claims in the turn.
        
        Returns a list of verdicts, one for each claim.
        Each verdict is one of: 'conflict', 'plausible', 'non-affirmative', 'nei'
        """
        # Build the turn context string
        main_qa = f"Main Question: {turn_data['main_question']}\nMain Response: {turn_data['main_response']}\n\n"
        
        # Build claims context - enumerate each claim individually
        claims_context = "Claims and Search Results:\n\n"
        claim_index = 1
        for sr in turn_data['search_results']:
            # Split claims by semicolon if multiple claims exist
            claims_list = [c.strip() for c in sr['claims'].split(';') if c.strip() and c.strip() != 'No claims']
            if not claims_list:
                claims_list = [sr['claims']]  # Use as-is if no semicolon separation
            
            for claim in claims_list:
                claims_context += f"=== Claim {claim_index} ===\n"
                claims_context += f"Claim: {claim}\n"
                claims_context += f"Search Output: {sr['output']}\n"
                if sr.get('confirmation_question') and sr.get('confirmation_response'):
                    claims_context += f"Confirmation Q: {sr['confirmation_question']}\n"
                    claims_context += f"Confirmation A: {sr['confirmation_response']}\n"
                claims_context += "================================\n\n"
                claim_index += 1
        
        user_content = main_qa + claims_context 

        if log_prompt:
            logging.info(f"[External Turn Verdict] System Prompt:\n{self.external_eval_prompt}")
            logging.info(f"[External Turn Verdict] User Prompt:\n{user_content}")
        
        # Create dynamic Pydantic model for the response
        class ClaimVerdict(BaseModel):
            claim_index: int = Field(..., description="Index of the claim (1-based)")
            claim: str = Field(..., description="The claim being evaluated")
            verdict: Literal['conflict', 'plausible', 'non-affirmative', 'nei'] = Field(..., description="The verdict for this claim")
            rationale: str = Field(..., description="The rationale behind the verdict")
        
        class ExternalTurnVerdictResponse(BaseModel):
            verdicts: List[ClaimVerdict] = Field(..., description="List of verdicts for each claim")
        
        completion_kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": self.external_eval_prompt},
                {"role": "user", "content": user_content}
            ],
            reasoning_effort="low",
            response_format=ExternalTurnVerdictResponse
        )
        
        if self.model.startswith("hosted_vllm/"):
            assert self.port is not None, "Port must be specified for hosted_vllm models."    
            completion_kwargs['api_base'] = f"http://{self.host}:{self.port}/v1"
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = get_completion(**completion_kwargs)
                self._calculate_cost(response)
                raw_content = response.choices[0].message.content

                if raw_content is None:
                    logging.warning(f"API returned None. Finish reason: {response.choices[0].finish_reason}")
                    raise ValueError("API returned None content")
                
                parsed_response = ExternalTurnVerdictResponse.model_validate_json(raw_content)
                return parsed_response
            except (ValidationError, ValueError) as e:
                if attempt == max_retries - 1:
                    logging.error(f"Pydantic validation failed after {max_retries} attempts: {e}")
                    raise
                logging.warning(f"Pydantic validation failed for ExternalTurnVerdictResponse, retrying... ({attempt + 1}/{max_retries})")
                continue
    
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
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = get_completion(**completion_kwargs)
                self._calculate_cost(response)
                parsed_response = InternalVerdictResponseFormat.model_validate_json(response.choices[0].message.content)
                return parsed_response
            except ValidationError as e:
                if attempt == max_retries - 1:
                    logging.error(f"Pydantic validation failed after {max_retries} attempts: {e}")
                    raise
                logging.warning(f"Pydantic validation failed for InternalVerdictResponseFormat, retrying... ({attempt + 1}/{max_retries})")
                continue
    
    def _extract_tool_output_str(self, tool_output: 'ToolOutput') -> str:
        """Extract tool output string including claims and output."""
        claims_str = ""
        if tool_output.arguments and 'claims' in tool_output.arguments:
            claims = tool_output.arguments['claims']
            if claims:
                claims_str = "Claims: " + "; ".join(claims) + "\n"
        output_str = f"Output: {tool_output.output}"
        return claims_str + output_str
    
    def _extract_search_result_info(self, tool_output: 'ToolOutput', agent_actions: List[Action] = None) -> Dict[str, Any]:
        """Extract search result info for external evaluation.
        
        Claims are stored in agent_action.tool_call.arguments, not in tool_output.arguments.
        We need to match by tool_call_id to find the corresponding claims.
        """
        claims = []
        
        # First try to get claims from agent_actions using tool_call_id matching
        if agent_actions and tool_output.tool_call_id:
            for action in agent_actions:
                if (action.tool_call and 
                    action.tool_call.details and 
                    action.tool_call.details.get('tool_calls')):
                    for tc in action.tool_call.details['tool_calls']:
                        if tc.get('id') == tool_output.tool_call_id:
                            # Found matching tool call, get claims from action.tool_call.arguments
                            if action.tool_call.arguments and 'claims' in action.tool_call.arguments:
                                claims = action.tool_call.arguments['claims'] or []
                            break
        
        # Fallback to tool_output.arguments if no claims found from agent_actions
        if not claims and tool_output.arguments and 'claims' in tool_output.arguments:
            claims = tool_output.arguments['claims'] or []
        
        return {
            'claims': "; ".join(claims) if claims else "No claims",
            'output': tool_output.output or "No output"
        }

    def consistency_eval(self, history: List[Turn]):
        """
        Evaluate consistency using history directly instead of evaluator's memory.
        
        For every answer from user:
        1. Check if the answer is uncooperative
        2. Internal check: previous Q&A (without tool outputs) + current Q&A
        3. External check: For turns with web search, evaluate each search result
        """
        # Filter out repeat turns
        non_repeat_turns = [turn for turn in history if turn.type != 'repeat']
        
        if not non_repeat_turns:
            return  # nothing to evaluate
        
        # Build evaluation items from history (for internal consistency)
        eval_items = []
        qa_history = "Previous Q&A:\n\n"  # For internal check (Q&A only, no tool outputs)
        
        # Build external evaluation items (turns with web search)
        external_eval_turns = []
        
        for turn_idx, turn in enumerate(non_repeat_turns):
            env_obs = turn.environment_observation
            if not env_obs:
                continue
            
            # Check if this turn has web search (agent: web_search in agent_action)
            has_web_search = False
            for action in turn.agent_action or []:
                if action.agent == "web_search":
                    has_web_search = True
                    break
            
            # First observation is always an interviewee response
            first_obs = env_obs[0]
            main_question = None
            main_response = None
            if first_obs.observation_type == "interviewee_response" and first_obs.response:
                main_question = first_obs.response.question
                main_response = first_obs.response.content
                
                # Add to eval items (for internal check)
                eval_items.append({
                    'turn_idx': turn_idx,
                    'question': main_question,
                    'response': main_response,
                    'qa_history': qa_history
                })
                
                # Update Q&A history
                qa_history += f"Q: {main_question}\nA: {main_response}\n\n"
            
            # Collect confirmation responses for internal evaluation (regardless of web_search)
            for obs in env_obs[1:]:
                if obs.observation_type == "interviewee_response" and obs.response:
                    # Add confirmation Q&A to eval_items for internal check
                    eval_items.append({
                        'turn_idx': turn_idx,
                        'question': obs.response.question,
                        'response': obs.response.content,
                        'qa_history': qa_history
                    })
                    qa_history += f"Q: {obs.response.question}\nA: {obs.response.content}\n\n"
            
            # If this turn has web search, build external eval data
            if has_web_search and main_question:
                # Collect tool outputs from this turn
                tool_outputs = []
                for obs in env_obs:
                    if obs.observation_type == "tool_output" and obs.tool_output:
                        tool_outputs.extend(obs.tool_output)
                
                # Collect confirmation responses for external evaluation
                confirmation_responses = []
                for obs in env_obs[1:]:
                    if obs.observation_type == "interviewee_response" and obs.response:
                        confirmation_responses.append(obs)
                
                # Build search results list for this turn
                search_results = []
                for i, tool_output in enumerate(tool_outputs):
                    sr_info = self._extract_search_result_info(tool_output, turn.agent_action)
                    sr_info['confirmation_question'] = None
                    sr_info['confirmation_response'] = None
                    if i < len(confirmation_responses):
                        sr_info['confirmation_question'] = confirmation_responses[i].response.question
                        sr_info['confirmation_response'] = confirmation_responses[i].response.content
                    search_results.append(sr_info)
                
                if search_results:
                    external_eval_turns.append({
                        'turn_idx': turn_idx,
                        'main_question': main_question,
                        'main_response': main_response,
                        'search_results': search_results
                    })
        
        if not eval_items:
            return  # nothing to evaluate
        
        # Step 1: Check uncooperative for ALL items
        qa_pairs = [{
            "question": item['question'], 
            "response": item['response'], 
            "log_prompt": (i < 3)
        } for i, item in enumerate(eval_items)]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            uncooperative_results = list(tqdm(executor.map(self.__generate_uncooperative_check, qa_pairs), 
                                              total=len(qa_pairs), 
                                              desc="Uncooperative check"))
        
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
        
        # Step 2: Internal consistency check (skip first item as it has no prior context)
        if len(eval_items) >= 2:
            consistency_eval_items = eval_items[1:]
            consistency_uncoop_results = uncooperative_results[1:]
            
            def generate_internal_wrapper(args):
                item, idx = args
                return self.__generate_internal_consistency_verdict(item['qa_history'], item['question'], item['response'], log_prompt=(idx < 3))
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                internal_results = list(tqdm(executor.map(generate_internal_wrapper, [(item, idx) for idx, item in enumerate(consistency_eval_items)]), 
                                            total=len(consistency_eval_items), 
                                            desc="Internal consistency check"))
            
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
        
        # Step 3: External consistency check (for turns with web search)
        total_search_results = 0
        non_affirmative_count = 0
        conflict_count = 0
        plausible_count = 0
        nei_count = 0
        
        if external_eval_turns:
            def generate_external_wrapper(args):
                turn_data, idx = args
                return self.__generate_external_turn_verdict(turn_data, log_prompt=(idx < 3))
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                external_results = list(tqdm(executor.map(generate_external_wrapper, [(turn, idx) for idx, turn in enumerate(external_eval_turns)]), 
                                            total=len(external_eval_turns), 
                                            desc="External consistency check"))
            
            # Process external results
            for turn_data, external_verdict in zip(external_eval_turns, external_results):
                turn_idx = turn_data['turn_idx']
                
                for verdict_item in external_verdict.verdicts:
                    total_search_results += 1
                    
                    detail = {
                        'turn_index': turn_idx,
                        'main_question': turn_data['main_question'],
                        'main_response': turn_data['main_response'],
                        'claim': verdict_item.claim,
                        'rationale': verdict_item.rationale
                    }
                    
                    if verdict_item.verdict == 'conflict':
                        conflict_count += 1
                        self.results_dict['external']['conflict']['count'] += 1
                        self.results_dict['external']['conflict']['details'].append(detail)
                    elif verdict_item.verdict == 'plausible':
                        plausible_count += 1
                        self.results_dict['external']['plausible']['count'] += 1
                        self.results_dict['external']['plausible']['details'].append(detail)
                    elif verdict_item.verdict == 'non-affirmative':
                        non_affirmative_count += 1
                        self.results_dict['external']['non-affirmative']['count'] += 1
                        self.results_dict['external']['non-affirmative']['details'].append(detail)
                    elif verdict_item.verdict == 'nei':
                        # NEI (No Evidence to evaluate) - not counted in consistency ratio
                        nei_count += 1
                        self.results_dict['external']['nei']['count'] += 1
                        self.results_dict['external']['nei']['details'].append(detail)
        
        # Calculate external scores
        # consistency_score = plausible / (total_claims - nei_claims)
        # NEI claims are excluded from consistency calculation
        evaluated_claims = conflict_count + plausible_count
        external_consistency_ratio = plausible_count / evaluated_claims if evaluated_claims > 0 else 0.0
        
        # affirmative_ratio = (total - non_affirmative) / total
        affirmative_ratio = (total_search_results - non_affirmative_count) / total_search_results if total_search_results > 0 else 0.0
        
        # Calculate final scores
        internal_plausible_ratio = (self.results_dict['internal']['plausible']['count'] / 
                                   (self.results_dict['internal']['conflict']['count'] + self.results_dict['internal']['plausible']['count'])) if (self.results_dict['internal']['conflict']['count'] + self.results_dict['internal']['plausible']['count']) > 0 else 0.0
        
        self._calculate_final_scores(responsive_ratio, affirmative_ratio, internal_plausible_ratio, external_consistency_ratio)
        
        # Save claims summary
        self.results_dict['external']['total_claims'] = total_search_results
    
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
            assert original.environment_observation[0].response.question in repeat.environment_observation[0].response.question, f"Mismatch in questions between original and repeat. Original: '{original.environment_observation[0].response.question}', Repeat: '{repeat.environment_observation[0].response.question}'."
            
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
        
        with ThreadPoolExecutor(max_workers=4) as executor:
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
            
        with ThreadPoolExecutor(max_workers=4) as executor:
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