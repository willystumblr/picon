import argparse
import re
import os
import json
import time
import glob
import logging
from dotenv import load_dotenv
from datasets import load_dataset
from litellm.cost_calculator import completion_cost

from src.utils import setup_logging, read_json, write_json, get_user_input_with_timeout, read_jsonl, get_completion
from src.env.interrogation_env import InterrogationEnv
from src.agents.agent_factory import get_agent
from src.schemas import Turn

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any


def parse_args():
    parser = argparse.ArgumentParser(description="Run the interrogation environment.")
    # Model selection
    parser.add_argument('--evaluator_model', type=str, default="gemini/gemini-2.5-flash", help='Model name for the evaluator.')
    # Port settings
    parser.add_argument('--evaluator_port', type=int, default=None, help='Port number for the evaluator agent server.')
    # Host settings
    parser.add_argument('--evaluator_host', type=str, default='localhost', help='Host for the evaluator agent server.')
   
    # Other configurations
    parser.add_argument('--log_to_file', action='store_true', help='Whether to log to a file.')
    parser.add_argument('--max_workers', type=int, default=5, help='Maximum number of workers for the interrogation.')
    
    # Input and output paths
    parser.add_argument('--evaluator_prompt_path', type=str, default='src/agents/prompts/evaluator_prompt.txt', help='Path to the evaluator agent prompt file.')
    parser.add_argument('--interview_data_dir', type=str, help='directory of interview data')
    return parser.parse_args()


   
def main(args, interview_path, result):
    results_complete = result
    #breakpoint()
    result_path = os.path.join(args.interview_data_dir, "evaluation", f"{os.path.basename(interview_path).split(".json")[0]}_Eval_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json")
    
    env = InterrogationEnv(
        agents = {
            "questioner": None,
            "extractor": None,
            "web_search": None,
            "evaluator": get_agent("evaluator", args.evaluator_prompt_path, model=args.evaluator_model, port=args.evaluator_port),
        },
        tools=None,
        max_turns=None,
        result_data = result,
    )
    env.agents['evaluator'].memory = result["agents_memory"]["evaluator"]

    first_history = [Turn(**h) for h in result['session_1']['history']]
    histories = [first_history]
    if 'session_2' in result:
        second_history = [Turn(**h) for h in result['session_2']['history']]
        histories.append(second_history)
        
    # Inter-session evaluation
    eval_result = env.evaluate(histories)
    results_complete["evaluation"]["model"] = args.evaluator_model
    results_complete["evaluation"]["cost"] = env.agents['evaluator'].cost
    results_complete["evaluation"] = eval_result
    
    write_json(results_complete, result_path)
    logging.info(f"Saved results to {result_path}.")        


if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_to_file=args.log_to_file, process_name="main")
    load_dotenv()
    
    interview_paths = glob.glob(os.path.join(args.interview_data_dir, "*.json"))
    evaluation_dir = os.path.join(args.interview_data_dir, "evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)
    
    proceed_list = []
    for interview_path in interview_paths:
        proceed_list.append((interview_path, json.load(open(interview_path, "r"))))
        
        
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(main, args, interview_path, interview_result): (interview_path, interview_result) for interview_path, interview_result in proceed_list}
        for future in as_completed(futures):
            interviewee_kwarg = futures[future]
            try:
                future.result()
            except Exception as e:
                logging.exception(f"Unhandled exception for interviewee {interview_path}: {e}")
        
    logging.info("All sessions completed.")