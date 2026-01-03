from src.utils import setup_logging, read_json, write_json, get_user_input_with_timeout, read_jsonl, get_completion
from src.env.interrogation_env import InterrogationEnv
from src.env.evaluator_test_env import EvaluatorTestEnv
from src.agents.agent_factory import get_agent
from src.tools.web_search import GoogleClaimSearch
from src.tools.address_locator import GoogleGeocodeValidate
from dotenv import load_dotenv
from datasets import load_dataset
from litellm.cost_calculator import completion_cost
import argparse
import re
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any

INTER_SESSION_PROMPT = """You will be given a single question and two or more corresponding answers from different sessions. Determine whether the answers are essentially the same in meaning.
If they are, output TRUE. If they are not, output FALSE.
Do not output any additional explanation or text."""

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

def inter_session_eval(args, env_lists: List[InterrogationEnv]) -> Dict[str, Any]:
    """
    Evaluate inter-session consistency by comparing 'get_to_know' responses across sessions.
    Returns a dictionary with inter-session evaluation results.
    """
    if len(env_lists) < 2:
        logging.warning("Inter-session evaluation requires at least 2 sessions. Skipping.")
        return {"inter_session_score": None, "inter_session_results": []}
    
    # Gather get_to_know data from each session
    inter_session_data = {}
    for session_idx, env in enumerate(env_lists):
        session_id = f"session_{session_idx + 1}"
        get_to_knows = [
            turn.environment_observation[0].response 
            for turn in env.state.history 
            if turn.type == 'get_to_know'
        ]
        inter_session_data[session_id] = get_to_knows
    
    # Ensure all sessions have the same number of get_to_know messages
    num_get_to_knows = min([len(v) for v in inter_session_data.values()])
    assert all([len(msgs) >= num_get_to_knows for msgs in inter_session_data.values()]), \
        "All sessions must have the same number of get-to-know messages for inter-session evaluation."
    
    evaluator_model = args.evaluator_model
    evaluator_port = args.evaluator_port
    
    # Build comparison messages for each question
    completion_kwargs_list = []
    for i in range(num_get_to_knows):
        # Build content with question and responses from all sessions
        first_session_id = list(inter_session_data.keys())[0]
        content = f"Question: {inter_session_data[first_session_id][i].question}\n"
        for session_id, get_to_knows in inter_session_data.items():
            content += f"Session `{session_id}` Response: {get_to_knows[i].content}\n"
        
        completion_kwargs = dict(
            model=evaluator_model,
            messages=[
                {"role": "system", "content": INTER_SESSION_PROMPT},
                {"role": "user", "content": content}
            ],
        )
        if evaluator_model.startswith("hosted_vllm/"):
            assert evaluator_port is not None, "Port must be specified for hosted_vllm models."
            completion_kwargs['api_base'] = f"http://localhost:{evaluator_port}/v1"
        completion_kwargs_list.append(completion_kwargs)
    
    # Run evaluations in parallel
    logging.info(f"Running inter-session evaluation for {len(completion_kwargs_list)} questions across {len(env_lists)} sessions.")
    with ThreadPoolExecutor(max_workers=min(8, len(completion_kwargs_list))) as executor:
        verdicts = list(executor.map(
            lambda kwargs: get_completion(**kwargs),
            completion_kwargs_list
        ))
    
    # Process results
    inter_session_results = []
    inter_session_scores = []
    total_cost = 0.0
    
    for i, verdict in enumerate(verdicts):
        total_cost += completion_cost(verdict) if not evaluator_model.startswith("hosted_vllm/") else 0.0
        judge = verdict.choices[0].message.content.strip() if verdict and verdict.choices and verdict.choices[0].message and verdict.choices[0].message.content else None
        
        # Retry if response is unexpected
        while judge not in ["TRUE", "FALSE"]:
            logging.warning(f"Unexpected response for inter-session eval: {judge}. Retrying...")
            verdict = get_completion(**completion_kwargs_list[i])
            total_cost += completion_cost(verdict) if not evaluator_model.startswith("hosted_vllm/") else 0.0
            judge = verdict.choices[0].message.content.strip() if verdict and verdict.choices and verdict.choices[0].message and verdict.choices[0].message.content else None
        
        is_consistent = (judge == "TRUE")
        inter_session_scores.append(is_consistent)
        
        first_session_id = list(inter_session_data.keys())[0]
        inter_session_results.append({
            "question": inter_session_data[first_session_id][i].question,
            "responses": {session_id: inter_session_data[session_id][i].content for session_id in inter_session_data.keys()},
            "is_consistent": judge,
        })
    
    inter_session_score = round(sum(inter_session_scores) / len(inter_session_scores), 4) if inter_session_scores else None
    
    logging.info(f"Inter-session evaluation completed. Score: {inter_session_score}, Cost: ${total_cost:.4f}")
    
    return {
        "inter_session_score": inter_session_score,
        "inter_session_results": inter_session_results,
        "inter_session_cost": total_cost
    }
    

def run_session(args, env: InterrogationEnv, reset_only=False):
    try:
        logging.info(f"Starting new session with interviewee: {env.interviewee.name}, baseline: {env.interviewee.type}")
        env.reset()
        if not reset_only:
            done = False
            while not done:
                state, done = env.step()
            env.finalize()
        result = env.save_state(reset_only=reset_only)
        return result, "Successfully completed"
    except Exception as e:
        logging.exception(f"Error during session with interviewee {env.interviewee.name}, baseline: {env.interviewee.type}: {e}")
        logging.info("Saving partial state...")
        termination_status=f"Error: {str(e)}"
        result = env.save_state(termination_status=termination_status)
        return result, termination_status

def main(args, interviewee_kwarg):
    results_complete = {}
    result_path = f"{args.output_dir}/{args.baseline_name}/{interviewee_kwarg.get('name', 'unknown').replace(' ', '_')}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    sessions = []
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
        sessions.append(env)
    
    # Inter-session evaluation
    inter_session_results = inter_session_eval(args, sessions)
    results_complete["inter_session_evaluation"] = inter_session_results
    
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