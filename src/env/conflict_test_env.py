import time
from src.env.interviewee_simulator import IntervieweeSimulator
from typing import List, Dict, Any
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

class ConflictTestEnv:
    def __init__(
        self, 
        model, 
        kg_path: str,
        **kwargs
        ):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.conflict_prompt = open(f"{project_root}/src/env/conflict_detection_prompt.txt", "r").read()
        self.model = model
        
        self.start_time = time.time()
        self.env_cost = 0.0
        self.kg_path = kg_path
        data = read_json(kg_path)
        self.qa_pairs = [item['environment_observation'][-1]['response'] for item in data['history'] if item['environment_observation'] and item['environment_observation'][-1]['observation_type'] == 'interviewee_response']
        self.kg = data.get('interviewee_kg', [])
        self.triplet_list_per_turn = [item["agent_action"][0]["content"] for item in data['history']]

        self.internal_conflict_cnt = 0
        self.internal_conflict_indices = []
        self.first_conflict_turn = False

    def reset(self):
        """reset the environment"""
        self.state = State(current_turn=1, history=[])
        return self.state
    
    def step(self): # Interviewee's response -> Extractor -> WebSearch (optional) -> Questioner -> Interviewee
        """run one turn of the interrogation"""
        if self.state.current_turn >= len(self.triplet_list_per_turn):
            logging.info("All triplets have been processed.")
            return self.state, True  # done
        
        curr_triplets = self.kg[self.state.current_turn]
        if not curr_triplets:
            logging.info(f"[TURN {self.state.current_turn}] No triplets extracted.")
            self.state.current_turn += 1
            return self.state, False
        logging.info(f"[TURN {self.state.current_turn}] Triplets extracted: {curr_triplets}")
        triplet_lists = self.triplet_list_per_turn[:self.state.current_turn]
        flattened_triplets = []
        for triplet_list in triplet_lists:
            flattened_triplets.extend(triplet_list)
        flattened_triplets = [str(triplet) for triplet in flattened_triplets]
        content = "Triplet List : \n" + "\n".join(flattened_triplets)
        message = [
            {
                "role": "system",
                "content": self.conflict_prompt
            },
            {
                "role": "user",
                "content": content
            }
        ]
        logging.info("Using sequential processing for conflict pair evaluation.")
        logging.info(f"Total triplets: {len(flattened_triplets)}")
        logging.info(f"Using model: {self.model}")
        logging.info("This may take a while...")
        res = get_completion(model=self.model, messages=message)
        if not res or not res.choices or not res.choices[0].message or not res.choices[0].message.content or res.choices[0].message.content.strip() not in ['plausible', 'conflict']:
            logging.warning(f"Unexpected response: {res}")
        else:
            self.env_cost += completion_cost(res)
            verdict = res.choices[0].message.content.lower()
            if "conflict" in verdict:
                self.internal_conflict_cnt += 1
                self.internal_conflict_indices.append(self.state.current_turn)
                logging.info(f"[CONFLICT DETECTION] Conflict detected at turn {self.state.current_turn}.")
        self.state.current_turn += 1
        return self.state, False
        
    def save_state(self, path: str, termination_status: str = "Successfully completed"):
        """save the current state to a json file"""
        final_result={
            "source_data_path": self.kg_path,
            "total_cost": self.env_cost,
            "duration": f"{(time.time() - self.start_time)/60} min", # in minutes
            "termination_status": termination_status,
            "history": [obj.model_dump() for obj in self.state.history],
            "internal_consistency": {
                "first_conflict_turn": self.first_conflict_turn,
                "conflict_pairs_count": self.internal_conflict_cnt,
                "conflict_turns": self.internal_conflict_indices,
                "conflict_triplets_accumulated":{
                    idx: self.triplet_list_per_turn[:idx+1] for idx in self.internal_conflict_indices
                }
                },
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
    parser.add_argument("--kg_path", type=str, required=True, help="Path to interview data JSON file")
    args = parser.parse_args()

    env = ConflictTestEnv(
        model=args.model,
        kg_path=args.kg_path
    )
    state = env.reset()
    done = False
    while not done:
        state, done = env.step()
    env.save_state(f"data/prompt_engineering/conflict_detection/conflict_detection_test_history_{time.strftime('%Y%m%d_%H%M%S')}.json")