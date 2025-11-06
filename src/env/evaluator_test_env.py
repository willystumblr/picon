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

class EvaluatorTestEnv:
    def __init__(
        self, 
        model, 
        interview_path: str,
        **kwargs
        ):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        # self.agent = get_agent("evaluator", model=model, system_message_path=f"{project_root}/src/env/evaluator_prompt.txt")
        self.model = model
        self.start_time = time.time()
        self.env_cost = 0.0
        self.interview_path = interview_path
        data = read_json(interview_path)
        self.evaluator_history = data["agent_memory"]["evaluator"]
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
        
        system_message_path = f"{project_root}/src/agents/prompts/evaluator_prompt.txt"
        self.system_prompt = open(system_message_path).read()
        self.evaluator_history[0]['content'] = self.system_prompt
        
    def reset(self):
        """reset the environment"""
        # self.state = State(current_turn=1, history=[]) # n-th turn indicates the n-th user response
        #breakpoint()
        self.user_indices = [i for i in range(0, len(self.evaluator_history)) if self.evaluator_history[i]['role']=='user'] # every 3 turns correspond to one user response
        self.user_indices.pop(0)  # remove the first user input which is the initial question

    
    def score_conflict(self, idx, verdict_action: Action):
        if verdict_action.content['verdict'] == 'conflict': # verdict_action.content['ground'] == 'internal':
            if verdict_action.content['ground'] == 'internal':
                self.internal_count += 1
                self.internal_conflict += 1
                self.internal_conflict_verdicts.append({
                    "turn": 0,#self.state.current_turn,
                    "verdict": verdict_action.content
                })
            else:
                self.external_count += 1
                self.external_conflict += 1
                self.external_conflict_verdicts.append({
                    "turn": 0,#self.state.current_turn,
                    "verdict": verdict_action.content
                })
            # if self.first_conflict_turn is None:
            #     # find the corresponding user response turn
            #     turn_idx = next(i for i, turn in enumerate(self.history) if turn['environment_observation'][0]['response'] == self.evaluator_history[idx]['content'])
            #     self.first_conflict_turn = turn_idx
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
            
        verdict = get_completion(
            model=self.model,
            messages=messages[:-1] + [{'role' : messages[-1]['role'] , 'content': messages[-1]['content'] + f"[conflict type] ground: {ground}"}], 
            temperature=1.0,
            response_format=EvaluationResponse
        )
        return verdict
    
    def step(self): # Interviewee's response -> Extractor -> WebSearch (optional) -> Questioner -> Interviewee
        """run one turn of the interrogation: process all at once"""
        messages_list = [
            self.evaluator_history[:i+1] for i in self.user_indices
        ]
        logging.info(f"[EVALUATOR] Evaluating {len(messages_list)} user responses for internal consistency.")
        #breakpoint()
        with ThreadPoolExecutor(max_workers=16) as executor:
            verdicts = list(tqdm(executor.map(
                lambda messages: self.generate_verdict(messages),
                messages_list
            )))
        
        for idx, verdict in enumerate(verdicts):
            self.env_cost += completion_cost(verdict)
            try:
                response = EvaluationResponse.model_validate_json(verdict.choices[0].message.content)
            except Exception as e:
                logging.error(f"[EVALUATOR] Validation error at index {idx}: {e}")
                logging.info(f"[EVALUATOR] Raw response: {verdict.choices[0].message.content}")
                breakpoint()
                
            content = response.model_dump()
            logging.info(f"[EVALUATOR] Verdict: {content['verdict']}, Ground: {content['ground']}, Rationale: {content['rationale']}")
            verdict_action = Action(
                agent="evaluator",
                action_type="respond",
                content=content
            )
            self.score_conflict(idx, verdict_action)
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
            }
        }
            
        write_json(final_result, path)
        
        logging.info(f"Saving final result to {path}")
        logging.info(f"Total cost: ${final_result['total_cost']}, Duration: {final_result['duration']}")
        

if __name__ == "__main__":
    from src.utils import setup_logging
    from argparse import ArgumentParser
    setup_logging(log_to_file=True, process_name="test_env")
    load_dotenv()

    parser = ArgumentParser(description="Questioner Test Environment")
    parser.add_argument("--model", type=str, default="gpt-5", help="Model to use")
    parser.add_argument("--interview_path", type=str, required=True, help="Path to interview data JSON file")
    args = parser.parse_args()

    env = EvaluatorTestEnv(
        model=args.model,
        interview_path=args.interview_path
    )
    state = env.reset()
    env.step()
    env.save_state(f"data/prompt_engineering/conflict_detection/evaluation_{time.strftime('%Y%m%d_%H%M%S')}.json")