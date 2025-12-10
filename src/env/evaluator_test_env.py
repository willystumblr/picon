import time
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal
from src.agents.base_agent import Agent
from src.agents.agent_factory import get_agent
from src.schemas import State, Action, Observation, Turn, ToolOutput, IntervieweeResponse
from src.utils import read_json, write_json, get_completion
import logging
from litellm.cost_calculator import completion_cost
import os
from tqdm import tqdm
from dotenv import load_dotenv
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor

class EvaluationResponse(BaseModel):
    verdict: Literal['conflict', 'plausible']  # 'conflict' or 'plausible'
    rationale: str  # explanation for the verdict
    ground: Literal['internal', 'external'] = Field(description="Ground for the verdict. If `internal`, the verdict is based on the internal context (i.e., the conversation history without external information). If `external`, it is based on external web search results.")

REPEAT_PROMPT = """You will be given a single question and two or more corresponding answers. Determine whether the the answers are essentially the same in meaning.
If they are, output TRUE. If they are not, output FALSE.
Do not output any additional explanation or text."""

class EvaluatorTestEnv:
    def __init__(
        self, 
        model: str, 
        interview_path: str,
        **kwargs
        ):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        # self.agent = get_agent("evaluator", model=model, system_message_path=f"{project_root}/src/env/evaluator_prompt.txt")
        self.model = model
        self.port = kwargs.get('port', None)
        self.start_time = time.time()
        self.env_cost = 0.0
        self.interview_path = interview_path
        all_data = read_json(interview_path) if isinstance(interview_path, str) else interview_path
        data = list(all_data.values())[0]  # first interview session per file
        self.num_sessions = len(all_data.keys())
        self.evaluator_history = data["agent_memory"]["evaluator"]
        self.evaluator_history = [self.evaluator_history[i] for i in range(len(self.evaluator_history)) if self.evaluator_history[i-1]!=self.evaluator_history[i] or i==0]
        self.history = data['history']
        
        self.repeat_results = data['repeat'].get('repeat_results', [])
        self.repeat_score = [(res["is_repeat"]=="TRUE") for res in self.repeat_results].count(True) / len(self.repeat_results) if len(self.repeat_results) > 0 else None
        
        self.internal_count = 0
        self.external_count = 0
        self.internal_conflict = 0
        self.external_conflict = 0
        self.internal_plausible = 0
        self.external_plausible = 0
        self.internal_conflict_verdicts = []
        self.external_conflict_verdicts = []
        self.first_conflict_turn = None
        self.all_verdicts = []
        
        self.inter_session_data = {}
        for session_id, d in all_data.items():
            if not session_id.startswith('session_'):
                continue
            get_to_knows = [turn["environment_observation"][0]['response'] for turn in d['history'] if turn['type'] == "get_to_know"]
            self.inter_session_data[session_id] = get_to_knows
        # gather each question-response pair across sessions
        self.inter_session_results = []
        self.inter_session_score = None

        self.abstention_prompt = open(f"{current_dir}/abstain_analysis_prompt.txt", "r").read()
        self.abstention_rate = 0.0
        self.abstention_results = []
        
    def reset(self):
        """reset the environment"""
        # self.state = State(current_turn=1, history=[]) # n-th turn indicates the n-th user response
        self.user_indices = [i for i in range(0, len(self.evaluator_history)) if self.evaluator_history[i]['role']=='user'] # every 3 turns correspond to one user response
        self.user_indices.pop(0)  # remove the first user input which is the initial question

    def _find_turn_idx(self, idx) -> int:
        """find the turn index in self.history corresponding to the idx-th user response in self.evaluator_history"""
        turn_idx = None
        user_response = self.evaluator_history[idx]['content']
        for i, turn in enumerate(self.history):
            if turn['type'] != 'repeat':
                for env_obs in turn['environment_observation']:
                    if env_obs["observation_type"] == "interviewee_response":
                        if env_obs["response"]["content"] == user_response:
                            turn_idx = i
                            break
        return turn_idx

    def score_conflict(self, idx, verdict_action: Action):
        self.all_verdicts.append({
            "message_idx": idx,
            "verdicts": {'value': verdict_action.content['verdict']}
        })
        if verdict_action.content['verdict'] == 'conflict': # verdict_action.content['ground'] == 'internal':
            turn_idx = self._find_turn_idx(idx)
            question = self.history[turn_idx]['environment_observation'][0]['response']['question']
            response = self.history[turn_idx]['environment_observation'][0]['response']['content']
            logging.info(f"[EVALUATOR] Conflict detected at turn {turn_idx} for Question: {question}, Response: {response}")
            if verdict_action.content['ground'] == 'internal':
                self.internal_count += 1
                self.internal_conflict += 1
                self.internal_conflict_verdicts.append({
                    "turn": turn_idx,
                    "question": question,
                    "response": response,
                    "verdict": verdict_action.content,
                })
            else:
                self.external_count += 1
                self.external_conflict += 1
                self.external_conflict_verdicts.append({
                    "turn": turn_idx,
                    "question": question,
                    "response": response,
                    "verdict": verdict_action.content,
                })
            if self.first_conflict_turn is None:
                self.first_conflict_turn = turn_idx
        else:
            if verdict_action.content['ground'] == 'internal':
                self.internal_count += 1
                self.internal_plausible += 1
            else:
                self.external_count += 1
                self.external_plausible += 1
    
    def generate_verdict(self, messages: List[Dict[str, Any]]) -> Action:
        #print(messages[-3]['role'])
        if messages[-3]['role']=='tool':
            ground = 'external'
        else:
            ground = 'internal'
        completion_kwargs = dict(
            model=self.model,
            messages=messages[:-1] + [{'role' : messages[-1]['role'] , 'content': messages[-1]['content'] + f"[conflict type] ground: {ground}"}], 
            temperature=1.0,
            response_format=EvaluationResponse
        )
        if self.model.startswith("hosted_vllm/"):
            assert self.port is not None, "Port must be specified for hosted_vllm models."    
            completion_kwargs['api_base'] = f"http://localhost:{self.port}/v1"    
        verdict = get_completion(**completion_kwargs)
        self.env_cost += completion_cost(verdict) if not self.model.startswith("hosted_vllm/") else 0.0
        return verdict
    
    def abstention_eval(self):
        evaluator_mem = self.evaluator_history
        user_resp = [(i, item) for i, item in enumerate(evaluator_mem) if item['role']=='user']
        qa_pairs = []
        for idx, user_item in user_resp:
            assistant_question = evaluator_mem[idx-1]['content']
            user_answer = user_item['content']
            qa_pairs.append({
                "question": assistant_question,
                "content": user_answer
            })
        class OutputSchema(BaseModel):
            abstain: Literal['true', 'partially true', 'false'] = Field(..., description="Whether the provided response is abstaining from answering the question.")
            reason : str = Field(..., description="A brief explanation for the abstention decision.")
            abstain_type: Literal['none', 'refusal', 'lack info', 'asking back', 'unrelated'] = Field(..., description="The type of abstention (e.g., 'lack info', 'unknown', etc.).")

        logging.info(f"Evaluating {len(qa_pairs)} QA pairs...")
        completion_kwargs_list = []
        for qa in qa_pairs:
            completion_kwargs = dict(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.abstention_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Question: {qa['question']}\nAnswer: {qa['content']}"
                    }
                ],
                response_format=OutputSchema,
            )
            if self.model.startswith("hosted_vllm/"):
                assert self.port is not None, "Port must be specified for hosted_vllm models."    
                completion_kwargs['api_base'] = f"http://localhost:{self.port}/v1"
            completion_kwargs_list.append(completion_kwargs)
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(
                lambda kwargs: get_completion(**kwargs),
                completion_kwargs_list
            ), total=len(qa_pairs), desc="Evaluating QA pairs"))
        self.env_cost += sum(completion_cost(res) for res in results) if not self.model.startswith("hosted_vllm/") else 0.0
        logging.info("Evaluation completed.")
    
        outputs = [OutputSchema.model_validate_json(res.choices[0].message.content) if res.choices[0].message.content else None for res in results]
        total_abstain_count = sum(1 for res in outputs if res and res.abstain!='false')    
        print(f"Total abstentions: {total_abstain_count} out of {len(qa_pairs)}")
        self.abstention_rate = total_abstain_count / len(qa_pairs)

        for i, res in enumerate(results):
            content = res.choices[0].message.content
            parsed_content = OutputSchema.model_validate_json(content)  # Validate the response
            
            self.abstention_results.append({
                "question": qa_pairs[i]['question'],
                "answer": qa_pairs[i]['content'],
                "abstain": parsed_content.abstain,
                "reason": parsed_content.reason,
                "abstain_type": parsed_content.abstain_type
            })



    def step(self): # Interviewee's response -> Extractor -> WebSearch (optional) -> Questioner -> Interviewee
        """run one turn of the interrogation: process all at once"""
        messages_list = [
            self.evaluator_history[:i+1] for i in self.user_indices
        ]
        logging.info(f"[EVALUATOR] Evaluating {len(messages_list)} user responses for consistency.")
        with ThreadPoolExecutor(max_workers=16) as executor:
            verdicts = list(tqdm(executor.map(
                lambda messages: self.generate_verdict(messages),
                messages_list
            )))
        
        for idx, verdict in zip(self.user_indices, verdicts):
            try:
                response = EvaluationResponse.model_validate_json(verdict.choices[0].message.content)
            except Exception as e:
                logging.error(f"[EVALUATOR] Validation error at index {idx}: {e}")
                logging.info(f"[EVALUATOR] Raw response: {verdict.choices[0].message.content}")
                
            content = response.model_dump()
            # logging.info(f"[EVALUATOR] Verdict: {content['verdict']}, Ground: {content['ground']}, Rationale: {content['rationale']}")
            verdict_action = Action(
                agent="evaluator",
                action_type="respond",
                content=content
            )
            self.score_conflict(idx, verdict_action)
        # breakpoint()
        # inter-session consistency
        # for each element pair/triplet/whatsoever in all_data.values() with same index across sessions, 
        # i.e. get_to_knows[0] in session 1, get_to_knows[0] in session 2, get_to_knows[0] in session 3, ...
        if self.num_sessions > 1:
            inter_session_messages_list = []
            num_get_to_knows = min([len(v) for v in self.inter_session_data.values()])
            for i in range(num_get_to_knows):
                # question, responses
                content = f"Question: {self.inter_session_data[list(self.inter_session_data.keys())[0]][i]['question']}" + "\n"
                for session_id, get_to_knows in self.inter_session_data.items():
                    content += f"Session `{session_id}` Response: {get_to_knows[i]['content']}\n"
                inter_session_messages_list.append([
                    {
                        "role": "system",
                        "content": REPEAT_PROMPT
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ])
            completion_kwargs_list = []
            for messages in inter_session_messages_list:
                completion_kwargs = dict(
                    model=self.model,
                    messages=messages,
                )            
                if self.model.startswith("hosted_vllm/"):
                    assert self.port is not None, "Port must be specified for hosted_vllm models."    
                    completion_kwargs['api_base'] = f"http://localhost:{self.port}/v1"    
                completion_kwargs_list.append(completion_kwargs)
            with ThreadPoolExecutor(max_workers=8) as executor:
                inter_session_verdicts = list(tqdm(executor.map(
                    lambda kwargs: get_completion(**kwargs),
                    completion_kwargs_list
                ), total=len(completion_kwargs_list), desc="Inter-session consistency evaluation"))
            inter_session_scores = []
            for i, verdict in enumerate(inter_session_verdicts):
                self.env_cost += completion_cost(verdict) if not self.model.startswith("hosted_vllm/") else 0.0
                judge = verdict.choices[0].message.content.strip() if verdict and verdict.choices and verdict.choices[0].message and verdict.choices[0].message.content else None
                inter_session_scores.append(judge == "TRUE")
                self.inter_session_results.append({
                    "question": self.inter_session_data[list(self.inter_session_data.keys())[0]][i]['question'],
                    "responses": {session_id: self.inter_session_data[session_id][i]['content'] for session_id in self.inter_session_data.keys()},
                    "is_consistent": judge,
                })

            self.inter_session_score = round(sum(inter_session_scores) / len(inter_session_scores), 4) if len(inter_session_scores) > 0 else None
        
        self.abstention_eval()

        return True
    
    
    def save_state(self, path: str, termination_status: str = "Successfully completed"):
        """save the current state to a json file"""
        final_result = {
            "interview_path": self.interview_path,
            "total_cost": self.env_cost,
            "duration": f"{(time.time() - self.start_time)/60:.2f} minutes",
            "first_conflict_turn": self.first_conflict_turn,
            "external_consistency":{
                "total_evaluations": self.external_count,
                "conflict_count": self.external_conflict,
                "plausible_count": self.external_plausible,
                "consistency_rate": (self.external_plausible / self.external_count) if self.external_count > 0 else None,
                "conflict_verdicts": self.external_conflict_verdicts
            },
            "internal_consistency":{
                "total_evaluations": self.internal_count,
                "conflict_count": self.internal_conflict,
                "plausible_count": self.internal_plausible,
                "consistency_rate": (self.internal_plausible / self.internal_count) if self.internal_count > 0 else None,
                "conflict_verdicts": self.internal_conflict_verdicts
            },
            "repeat_score":{
                "repeat_score": self.repeat_score,
                "repeat_results": self.repeat_results
            },
            "inter_session_score": {
                "inter_session_score": self.inter_session_score,
                "inter_session_results": self.inter_session_results
            },
            "abstention_analysis": {
                "abstention_rate": self.abstention_rate,
                "abstention_results": self.abstention_results
            },
        }
            
        write_json(final_result, path)
        
        logging.info(f"Saving final result to {path}")
        logging.info(f"Total cost: ${final_result['total_cost']}, Duration: {final_result['duration']}")
        logging.info("Results summary:")
        logging.info(f"External Consistency Rate: {final_result['external_consistency']['consistency_rate']}")
        logging.info(f"Internal Consistency Rate: {final_result['internal_consistency']['consistency_rate']}")
        logging.info(f"Inter-Session Consistency Rate: {final_result['inter_session_score']['inter_session_score']}")
        logging.info(f"Repeat Score: {final_result['repeat_score']['repeat_score']}")
        return 

if __name__ == "__main__":
    from src.utils import setup_logging
    from argparse import ArgumentParser
    setup_logging(log_to_file=True, process_name="test_env")
    load_dotenv()

    parser = ArgumentParser(description="Evaluator Test Environment")
    parser.add_argument("--model", type=str, default="gpt-5", help="Model to use")
    parser.add_argument("--interview_path", type=str, required=True, help="Path to interview data JSON file")
    parser.add_argument("--baseline_name", type=str, required=True, help="Baseline name for saving results")
    args = parser.parse_args()

    env = EvaluatorTestEnv(
        model=args.model,
        interview_path=args.interview_path,
        port=8888
    )
    state = env.reset()
    env.step()
    env.save_state(f"data/evaluation/{args.baseline_name}/evaluation_{os.path.basename(args.interview_path)}")