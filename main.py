import os
from src.utils import setup_logging, read_json, write_json, get_user_input_with_timeout, read_jsonl
from src.env.interrogation_env import InterrogationEnv
from src.env.evaluator_test_env import EvaluatorTestEnv
from src.agents.agent_factory import get_agent
from src.tools.web_search import GoogleClaimSearch
from src.tools.address_locator import GoogleGeocodeValidate
from dotenv import load_dotenv
from datasets import load_dataset
import argparse
import re
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

def parse_args():
    parser = argparse.ArgumentParser(description="Run the interrogation environment.")
    # Model selection
    parser.add_argument('--baseline_name', type=str, required=True, help='Baseline name for the interviewee simulator.', choices=['characterai', 'human_simulacra', 'opencharacter', 'consistent_llm', 'human_interview', 'naive_human_simulacra'])
    parser.add_argument('--questioner_model', type=str, default="gemini/gemini-2.5-flash", help='Model name for the questioner.')
    parser.add_argument('--extractor_model', type=str, default="gemini/gemini-2.5-flash", help='Model name for the extractor.')
    parser.add_argument('--web_search_model', type=str, default="gemini/gemini-2.5-flash", help='Model name for the web search agent.')
    parser.add_argument('--evaluator_model', type=str, default="gemini/gemini-2.5-flash", help='Model name for the evaluator.')
    parser.add_argument('--simulator_model', type=str, help='Simulator model name (opencharacter & consistent_llm).')
    parser.add_argument('--nhd_model', type=str, default="gemini/gemini-2.5-flash", help='Model name for the NH detector in the interviewee simulator.')
    # Port settings
    parser.add_argument('--questioner_port', type=int, default=None, help='Port number for the questioner agent server.')
    parser.add_argument('--extractor_port', type=int, default=None, help='Port number for the extractor agent server.')
    parser.add_argument('--web_search_port', type=int, default=None, help='Port number for the web search agent server.')
    parser.add_argument('--evaluator_port', type=int, default=None, help='Port number for the evaluator agent server.')
    parser.add_argument('--simulator_port', type=int, default=None, help='Port number for the persona simulator.')
    parser.add_argument('--nhd_port', type=int, default=None, help='Port number for the NH detector in the interviewee simulator.')
    # Host settings
    parser.add_argument('--questioner_host', type=str, default='localhost', help='Host for the questioner agent server.')
    parser.add_argument('--extractor_host', type=str, default='localhost', help='Host for the extractor agent server.')
    parser.add_argument('--web_search_host', type=str, default='localhost', help='Host for the web search agent server.')
    parser.add_argument('--evaluator_host', type=str, default='localhost', help='Host for the evaluator agent server.')
    parser.add_argument('--simulator_host', type=str, default='localhost', help='Host for the persona simulator.')
    parser.add_argument('--nhd_host', type=str, default='localhost', help='Host for the NH detector in the interviewee simulator.')
    # Other configurations
    parser.add_argument('--num_turns', type=int, default=30, help='Maximum number of turns in the interrogation.')
    parser.add_argument('--num_sessions', type=int, default=2, help='Number of interrogation sessions to run per interviewee.')
    parser.add_argument('--max_workers', type=int, default=5, help='Maximum number of workers for the interrogation.')
    parser.add_argument('--do_sample', action='store_true', help='Whether to sample OpenCharacter personas.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling personas.')
    parser.add_argument('--question_seed', type=int, default=42, help='Random seed for pre-defined questions\' order.')
    parser.add_argument('--log_to_file', action='store_true', help='Whether to log to a file.')
    # Input and output paths
    parser.add_argument('--questioner_prompt_path', type=str, default='src/agents/prompts/questioner.txt', help='Path to the questioner agent prompt file.')
    parser.add_argument('--entity_extractor_prompt_path', type=str, default='src/agents/prompts/entity_extractor.txt', help='Path to the entity extractor agent prompt file.')
    parser.add_argument('--claim_extractor_prompt_path', type=str, default='src/agents/prompts/claim_extractor_prompt.txt', help='Path to the claim extractor agent prompt file.')
    parser.add_argument('--web_search_prompt_path', type=str, default='src/agents/prompts/websearch_prompt.txt', help='Path to the web search agent prompt file.')
    parser.add_argument('--evaluator_prompt_path', type=str, default='src/agents/prompts/evaluator_prompt.txt', help='Path to the evaluator agent prompt file.')
    parser.add_argument('--output_dir', type=str, default='data/results', help='Directory to save the results.')
    parser.add_argument('--temp_output_dir', type=str, default='data/temp_results', help='Directory to save temporary results in case of errors.')
    
    
    return parser.parse_args()

def run_session(args, env: InterrogationEnv, reset_only=False):
    try:
        logging.info(f"Starting new session with interviewee: {env.interviewee.name}, baseline: {env.interviewee.type}")
        env.reset()
        if not reset_only:
            done = False
            while not done:
                state, done = env.step()
            env.finalize()
        result = env.save_state()
        return result, "Successfully completed"
    except Exception as e:
        logging.exception(f"Error during session with interviewee {env.interviewee.name}, baseline: {env.interviewee.type}: {e}")
        logging.info("Saving partial state...")
        termination_status=f"Error: {str(e)}"
        result = env.save_state(termination_status=termination_status)
        return result, termination_status

def main(args, interviewee_kwarg):
    results_complete = {}
    result_path = f"{args.output_dir}/{args.baseline_name}/{interviewee_kwarg.get('name', 'unknown').replace(' ', '_')}_{args.questioner_model.split('/')[-1]}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    
    for session_idx in range(args.num_sessions):
        tools = {
            "google_claim_search": GoogleClaimSearch(
                api_key=os.getenv('GOOGLE_CLAIM_SEARCH'),
                cx=os.getenv('GOOGLE_CX_ID'),
            ),
            "google_geocode_validate": GoogleGeocodeValidate(api_key=os.getenv('GOOGLE_GEOCODE'))
        }
        env = InterrogationEnv(
            agents = {
                "questioner": get_agent("questioner", args.questioner_prompt_path, model=args.questioner_model, port=args.questioner_port),
                "extractor": get_agent("entity_extractor", args.entity_extractor_prompt_path, model=args.extractor_model, port=args.extractor_port),
                "web_search": get_agent("web_search", args.web_search_prompt_path, model=args.web_search_model, port=args.web_search_port),
                "evaluator": get_agent("evaluator", args.evaluator_prompt_path, model=args.evaluator_model, port=args.evaluator_port),
            },
            tools=tools,
            max_turns=args.num_turns,
            **interviewee_kwarg
        )
        logging.info(f"Starting session {session_idx + 1}/{args.num_sessions} for interviewee: {env.interviewee.name}, baseline: {env.interviewee.type}")
        reset_only = False
        if session_idx > 0:
            # reset only to start a new session
            logging.info("Resetting environment for new session...")
            reset_only = True
        session_result, status = run_session(args, env, reset_only=reset_only)
        results_complete[f"session_{session_idx + 1}"] = session_result
    write_json(results_complete, result_path)
    logging.info("Starting evaluation with EvaluatorTestEnv...")
    evaluator_env = EvaluatorTestEnv(
        model=args.evaluator_model,
        interview_path=results_complete,
        port=args.evaluator_port
    )
    evaluator_env.reset()
    evaluator_env.step()

    results_complete["fist_conflict_turn"] = evaluator_env.first_conflict_turn
    results_complete["external_consistency"] = {
        "total_evaluations": evaluator_env.external_count,
        "conflict_count": evaluator_env.external_conflict,
        "plausible_count": evaluator_env.external_plausible,
        "consistency_rate": (evaluator_env.external_plausible / evaluator_env.external_count) if evaluator_env.external_count > 0 else None,
        "conflict_verdicts": evaluator_env.external_conflict_verdicts
    }
    results_complete["internal_consistency"] = {
        "total_evaluations": evaluator_env.internal_count,
        "conflict_count": evaluator_env.internal_conflict,
        "plausible_count": evaluator_env.internal_plausible,
        "consistency_rate": (evaluator_env.internal_plausible / evaluator_env.internal_count) if evaluator_env.internal_count > 0 else None,
        "conflict_verdicts": evaluator_env.internal_conflict_verdicts
    }
    results_complete["inter_session_score"] = {
        "inter_session_score": evaluator_env.inter_session_score,
        "inter_session_results": evaluator_env.inter_session_results
    }
    results_complete["abstention_eval"] = {
        "abstention_rate": evaluator_env.abstention_rate,
        "abstention_results": evaluator_env.abstention_results
    }
    results_complete["eval_cost"] = evaluator_env.env_cost
    # results_complete["total_cost"] = evaluator_env.env_cost
    # for session in results_complete.values():
    #     results_complete["total_cost"] += session['cost']['total_cost']
    write_json(results_complete, result_path)
    logging.info(f"Saved results to {result_path}.")

if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_to_file=args.log_to_file, process_name="main")
    load_dotenv()
    
    interviewee_kwargs = []
    # set up baseline interviewee simulator
    if args.baseline_name == "characterai":
        assert os.getenv('CAI_API_KEY') is not None, "Character AI requires user_id parameter"
        personas = read_json("src/env/personas/characterai.json")
        for persona in personas:
            interviewee_kwargs.append({
                "baseline_name": "characterai",
                "character_id": persona['character_id'],
                "user_id": os.getenv('CAI_API_KEY'), #args.user_id,
                "name": persona['character_name'],
                "nhd_model": args.nhd_model,
                "question_seed": args.question_seed,
                "nhd_port": args.nhd_port,
            })    
    elif "human_simulacra" in args.baseline_name:
        interviewee_kwargs = [{
            "baseline_name": args.baseline_name,
            "name": name,            
            "nhd_model": args.nhd_model,
            "simulator_model": args.simulator_model,
            "question_seed": args.question_seed,
            "nhd_port": args.nhd_port,
        } for name in ["Mary Jones", "Haley Collins", "Sara Ochoa", "James Jones", "Tami Clark", "Michael Miller", "Kevin Kelly", "Erica Walker", "Leslie Nichols", "Robert Scott", "Marsh Zhaleh"]]
    elif args.baseline_name == "opencharacter":
        dataset = load_dataset("xywang1/OpenCharacter", "Synthetic-Character", split="train")
        if args.do_sample:
            dataset = dataset.shuffle(seed=args.seed).select(range(10))
        for data in dataset:
            name_match = re.match(r"Name:\s(.*)\n",  data['character'])
            if not name_match:
                logging.warning(f"Could not extract name from character profile: {data['character']}. Skipping this persona.")
                continue
            interviewee_kwargs.append({
                "baseline_name": "opencharacter",
                "model_path": "willystumblr/opencharacter-sft-2025-06-21_14-54-13", # hardcoded for now
                "persona": data['persona'],
                "profile": data['character'],
                "name": name_match.group(1).strip(),
                "load_in_4bit": True,
                "nhd_model": args.nhd_model,
                "question_seed": args.question_seed,
                "simulator_model": args.simulator_model,
                "port": args.simulator_port,
                "simulator_host": args.simulator_host,
                "nhd_port": args.nhd_port,
            })
    elif args.baseline_name == "consistent_llm":
        dataset = read_jsonl("src/env/personas/consistent_llm_personas.jsonl")
        if args.do_sample:
            import random
            random.seed(args.seed)
            dataset = random.sample(dataset, k=10)
        for data in dataset:
            interviewee_kwargs.append({
                "baseline_name": "consistent_llm",
                "model_path": "/home/edlab/sjim/consistent-LLMs/rl_training/checkpoints/chatting/llama-8b-sft-ppo-prompt",
                "persona": data['persona'],
                "name": data['name'],
                "counterpart_name": data['counterpart_name'],
                "instruction": data['instruction'],
                "nhd_model": args.nhd_model,
                "nhd_port": args.nhd_port,
                "question_seed": args.question_seed,
                "simulator_model": args.simulator_model,
                "simulator_host": args.simulator_host,
                "port": args.simulator_port
            })
    elif args.baseline_name == "persona_hub":
        interviewee_kwargs = [{
            "baseline_name": "persona_hub",
            "name": input("Enter your name: "),
            "nhd_model": args.nhd_model,
            "nhd_port": args.nhd_port,
            "question_seed": args.question_seed
        }]
        
    elif args.baseline_name == "human_interview":
        interviewee_kwargs = [{
            "baseline_name": "human_interview",
            "name": input("Enter your name: "),
            "nhd_model": args.nhd_model,
            "nhd_port": args.nhd_port,
            "question_seed": args.question_seed
        }]
    else:
        raise ValueError("Invalid baseline name. Choose from ['characterai', 'human_simulacra', 'opencharacter', 'human_interview']")
    
    proceed_list = []
    for interviewee_kwarg in interviewee_kwargs:
        logging.info(f"Proceed to the interview session? [Y/N] (Interviewee: {interviewee_kwarg.get('name', 'unknown')}, baseline: {interviewee_kwarg['baseline_name']})")
        # proceed if y or no input for 10 seconds, else skip
        while True:
            user_input = get_user_input_with_timeout(timeout=10)
            if not user_input or user_input.lower() == 'y':
                logging.info(f"Interviewee: {interviewee_kwarg.get('name', 'unknown')}, baseline: {interviewee_kwarg['baseline_name']} added to the queue.")
                proceed_list.append(interviewee_kwarg)
                break
            elif user_input.lower() == 'n':
                logging.info("Skipping this interviewee.")
                break
            else:
                logging.info("Invalid input. Please enter Y or N.")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(main, args, interviewee_kwarg): interviewee_kwarg for interviewee_kwarg in proceed_list}
        for future in as_completed(futures):
            interviewee_kwarg = futures[future]
            try:
                future.result()
            except Exception as e:
                logging.exception(f"Unhandled exception for interviewee {interviewee_kwarg.get('name', 'unknown')}, baseline: {interviewee_kwarg['baseline_name']}: {e}")
    logging.info("All sessions completed.")