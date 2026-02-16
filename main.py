from src.utils import setup_logging, read_json, write_json, get_user_input_with_timeout, read_jsonl, get_completion
from src.env.interrogation_env import InterrogationEnv
from src.agents.agent_factory import get_agent
from src.tools.web_search import GoogleClaimSearch
from src.tools.address_locator import GoogleGeocodeValidate
from dotenv import load_dotenv
import argparse
import re
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Run the interrogation environment.")
    # Model selection
    parser.add_argument('--baseline_name', type=str, required=True, help='Baseline name for the interviewee simulator.', choices=['characterai', 'human_simulacra', 'opencharacter', 'consistent_llm', 'human_interview', 'naive_human_simulacra', 'persona_hub', 'twin_2k_500', 'deeppersona', 'llm_generated'])
    parser.add_argument('--questioner_model', type=str, default="gemini/gemini-2.5-flash", help='Model name for the questioner.')
    parser.add_argument('--extractor_model', type=str, default="gemini/gemini-2.5-flash", help='Model name for the extractor.')
    parser.add_argument('--web_search_model', type=str, default="gemini/gemini-2.5-flash", help='Model name for the web search agent.')
    parser.add_argument('--evaluator_model', type=str, default="gemini/gemini-2.5-flash", help='Model name for the evaluator.')
    parser.add_argument('--simulator_model', type=str, help='Simulator model name (opencharacter, consistent_llm, llm_generated).')
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
    parser.add_argument('--question_file_path', type=str, default='src/env/wvs_orthogonal_questions.json', help='Path to the pre-defined questions file.')
    parser.add_argument('--eval_factors', type=str, nargs='+', default=None, choices=['internal', 'external', 'intra', 'inter'], help='Evaluation factors to compute. If not specified, all factors are evaluated.')
    parser.add_argument('--do_eval', action='store_true', help='Whether to run evaluation after interview sessions.')
    
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
        result = env.save_state(reset_only=reset_only)
        return result, "Successfully completed"
    except Exception as e:
        logging.exception(f"Error during session with interviewee {env.interviewee.name}, baseline: {env.interviewee.type}: {e}")
        logging.info("Saving partial state...")
        termination_status=f"Error: {str(e)}"
        result = env.save_state(termination_status=termination_status)
        return result, termination_status

def main(args, interviewee_kwarg):
    """Run interview sessions for a single persona and return stats for aggregation."""
    results_complete = {}
    result_path = f"{args.output_dir}/{args.baseline_name}/{interviewee_kwarg.get('name', 'unknown').replace(' ', '_')}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    
    # Initialize stats for this persona
    persona_stats = {
        "name": interviewee_kwarg.get('name', 'unknown'),
        "ai_detected": False,
        "success": False,
        "error_type": None,
        "duration_min": 0.0,
        "total_cost": 0.0,
        "agents_cost": 0.0,
        "interviewee_cost": 0.0,
        "tool_costs": 0.0,
        "num_interviewee_responses": 0,
        "num_turns_completed": 0,
        "num_tool_calls": 0,
        "sessions_completed": 0,
        # Evaluation scores (None if not evaluated)
        "eval_internal_harmonic_mean": None,
        "eval_internal_responsiveness": None,
        "eval_internal_consistency": None,
        "eval_external_wilson": None,
        "eval_stability_inter_session": None,
        "eval_stability_intra_session": None,
    }
    
    tools = {
            "google_claim_search": GoogleClaimSearch(
                api_key=os.getenv('GOOGLE_CLAIM_SEARCH'),
                cx=os.getenv('GOOGLE_CX_ID'),
            ),
            "google_geocode_validate": GoogleGeocodeValidate(api_key=os.getenv('GOOGLE_GEOCODE'))
        }
    
    try:
        env = InterrogationEnv(
            agents = {
                "questioner": get_agent("questioner", args.questioner_prompt_path, model=args.questioner_model, port=args.questioner_port),
                "extractor": get_agent("entity_extractor", args.entity_extractor_prompt_path, model=args.extractor_model, port=args.extractor_port),
                "web_search": get_agent("web_search", args.web_search_prompt_path, model=args.web_search_model, port=args.web_search_port),
                "evaluator": get_agent("evaluator", args.evaluator_prompt_path, model=args.evaluator_model, port=args.evaluator_port),
            },
            tools=tools,
            max_turns=args.num_turns,
            question_path=args.question_file_path,
            **interviewee_kwarg
        )
        
        reset_only = False
        histories = []
        for session_idx in range(args.num_sessions):
            logging.info(f"Starting session {session_idx + 1}/{args.num_sessions} for interviewee: {env.interviewee_kwargs['name']}, baseline: {env.baseline_name}")
            logging.info("Resetting environment for new session...")
            env.reset(reset_only=reset_only)
            if not reset_only:
                done = False
                while not done:
                    state, done = env.step()
                state = env.finalize()
            session_result = env.save_state()
            # Use env.state.history instead of local state variable for reset_only sessions
            histories.append(env.state.history)
            results_complete[f"session_{session_idx+1}"] = session_result
            logging.info(f"Completed session {session_idx + 1}/{args.num_sessions} for interviewee: {env.interviewee.name}, baseline: {env.interviewee.type}")
            persona_stats["sessions_completed"] += 1
            reset_only = True
        
        results_complete["agents_memory"] = {agent_name: agent.memory for agent_name, agent in env.agents.items()}
        write_json(results_complete, result_path)
        
        # Inter-session evaluation (only if --do_eval is set)
        if args.do_eval:
            eval_result = env.evaluate(histories, eval_factors=args.eval_factors)
            results_complete["evaluation"] = eval_result
            write_json(results_complete, result_path)
            
            # Extract evaluation scores for aggregation
            if eval_result:
                internal = eval_result.get("internal", {})
                external = eval_result.get("external", {})
                stability = eval_result.get("stability", {})
                
                internal_score = internal.get("score", {})
                persona_stats["eval_internal_harmonic_mean"] = internal_score.get("harmonic_mean")
                persona_stats["eval_internal_responsiveness"] = internal_score.get("responsiveness_score")
                persona_stats["eval_internal_consistency"] = internal_score.get("consistency_score")
                
                external_score = external.get("score", {})
                persona_stats["eval_external_wilson"] = external_score.get("wilson_score")
                
                inter_session = stability.get("inter_session", {})
                intra_session = stability.get("intra_session", {})
                persona_stats["eval_stability_inter_session"] = inter_session.get("score")
                persona_stats["eval_stability_intra_session"] = intra_session.get("score")
        
        logging.info(f"Saved results to {result_path}.")
        
        # Collect stats from completed sessions
        persona_stats["success"] = True
        for session_key in [k for k in results_complete.keys() if k.startswith("session_")]:
            session_data = results_complete[session_key]
            # Parse duration (stored as "X.XXX min")
            duration_str = session_data.get("duration", "0 min")
            try:
                persona_stats["duration_min"] += float(duration_str.replace(" min", ""))
            except:
                pass
            # Costs
            cost_data = session_data.get("cost", {})
            persona_stats["total_cost"] += cost_data.get("total_cost", 0.0)
            persona_stats["agents_cost"] += cost_data.get("agents_cost", 0.0)
            persona_stats["interviewee_cost"] += cost_data.get("interviewee_cost", 0.0)
            persona_stats["tool_costs"] += sum(cost_data.get("tool_costs", {}).values())
            # Count interviewee responses and tool calls from history
            history = session_data.get("history", [])
            for turn in history:
                for obs in turn.get("environment_observation", []):
                    if obs.get("observation_type") == "interviewee_response":
                        persona_stats["num_interviewee_responses"] += 1
                    if obs.get("observation_type") == "tool_output":
                        tool_outputs = obs.get("tool_output", [])
                        persona_stats["num_tool_calls"] += len(tool_outputs) if tool_outputs else 0
                # Count turns (main_interrogation type)
                if turn.get("type") == "main_interrogation":
                    persona_stats["num_turns_completed"] += 1
        
        return persona_stats
        
    except ValueError as e:
        if "AI Detected" in str(e):
            persona_stats["ai_detected"] = True
            persona_stats["error_type"] = "AI Detected"
            # For AI detected cases, inter-session score should be 0.0
            persona_stats["eval_stability_inter_session"] = 0.0
            logging.warning(f"AI Detected for persona {persona_stats['name']}")
        else:
            persona_stats["error_type"] = str(e)
        return persona_stats
    except Exception as e:
        persona_stats["error_type"] = str(e)
        # For failed cases, inter-session score should be 0.0
        persona_stats["eval_stability_inter_session"] = 0.0
        logging.exception(f"Error for persona {persona_stats['name']}: {e}")
        return persona_stats

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
        from datasets import load_dataset
        dataset = load_dataset("xywang1/OpenCharacter", "Synthetic-Character", split="train")
        if args.do_sample:
            dataset = dataset.shuffle(seed=args.seed).select(range(12))
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
            dataset = random.sample(dataset, k=15)
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
        #dataset = load_dataset("proj-persona/PersonaHub", "persona", split="train")
        dataset = read_jsonl("src/env/personas/persona_hub/named_personas_with_key.jsonl")
        # if args.do_sample:
        #     dataset = dataset.shuffle(seed=args.seed).select(range(10))
        if args.do_sample:
            import random
            random.seed(args.seed)
            dataset = random.sample(dataset, k=10)
        for data in dataset:
            data['persona'] = data['persona'][0].lower() + data['persona'][1:] if len(data['persona']) > 1 else data['persona'].lower()
            interviewee_kwargs.append({
                "baseline_name": "persona_hub",
                "persona": data['persona'],
                "name" : data['name'],
                "nhd_model": args.nhd_model,
                "nhd_port": args.nhd_port,
                "question_seed": args.question_seed,
                "simulator_model": args.simulator_model,
                "simulator_host": args.simulator_host,
                "port": args.simulator_port
            })
        
    elif args.baseline_name == "human_interview":
        interviewee_kwargs = [{
            "baseline_name": "human_interview",
            "name": input("Enter your name: "),
            "nhd_model": args.nhd_model,
            "nhd_port": args.nhd_port,
            "question_seed": args.question_seed
        }]
    elif args.baseline_name == "twin_2k_500":
        
        dataset = load_dataset("LLM-Digital-Twin/Twin-2K-500", "full_persona", split="data")
        if args.do_sample:
            dataset = dataset.shuffle(seed=args.seed).select(range(10))
        for data in dataset:
            name = f"Twin-{data['pid']}"
            interviewee_kwargs.append({
                "baseline_name": "twin_2k_500",
                "simulator_model": args.simulator_model,
                "persona": data['persona_json'],
                "name": name,
                "nhd_model": args.nhd_model,
                "question_seed": args.question_seed,
                "port": args.simulator_port,
                "simulator_host": args.simulator_host,
            })
    
    elif args.baseline_name == "deeppersona":
        import glob
        dataset_path = "/home/data_storage/deeppersona"
        persona_files = glob.glob(os.path.join(dataset_path, "*.json"))
        for persona_file in persona_files:
            data = read_json(persona_file)
            name = os.path.basename(persona_file).replace(".json", "")
            interviewee_kwargs.append({
                "baseline_name": "deeppersona",
                "simulator_model": args.simulator_model,
                "persona": data,
                "name": name,
                "nhd_model": args.nhd_model,
                "question_seed": args.question_seed,
                "port": args.simulator_port,
                "simulator_host": args.simulator_host,
            })
    elif args.baseline_name == "llm_generated":
        dataset = load_dataset("Tianyi-Lab/Personas", split="train")
        if args.do_sample:
            sample_size = min(10, len(dataset))
            dataset = dataset.shuffle(seed=args.seed).select(range(sample_size))

        # Tianyi-Lab/Personas stores persona text in model-specific columns.
        preferred_prefixes = [
            "Llama-3.1-70B-Instruct",
            # "Qwen2.5-72B-Instruct",
            # "Athene-70B",
            # "Mixtral-8x7B-Instruct-v0.1",
            # "Nemotron-70B-Instruct",
            # "Llama-3.1-8B-Instruct",
        ]
        available_prefixes = [
            col[: -len("_descriptive_persona")]
            for col in dataset.column_names
            if col.endswith("_descriptive_persona")
        ]

        selected_prefix = None
        for prefix in preferred_prefixes:
            if prefix in available_prefixes:
                selected_prefix = prefix
                break
        if selected_prefix is None and available_prefixes:
            selected_prefix = available_prefixes[0]
        if selected_prefix is None:
            raise ValueError(
                "No persona columns found in Tianyi-Lab/Personas. "
                "Expected *_descriptive_persona columns."
            )

        logging.info(f"Using persona columns from: {selected_prefix}")
        for data in dataset:
            persona_number = data.get("persona_number")
            persona = {
                "meta_persona": data.get("meta_persona", ""),
                "descriptive_persona": data.get(f"{selected_prefix}_descriptive_persona", ""),
                "objective_table_persona": data.get(f"{selected_prefix}_objective_table_persona", ""),
                "subjective_table_persona": data.get(f"{selected_prefix}_subjective_table_persona", ""),
            }
            interviewee_kwargs.append({
                "baseline_name": "llm_generated",
                "simulator_model": args.simulator_model,
                "persona": persona,
                "name": f"LLM-Persona-{persona_number}" if persona_number is not None else "LLM-Persona-unknown",
                "nhd_model": args.nhd_model,
                "question_seed": args.question_seed,
                "port": args.simulator_port,
                "simulator_host": args.simulator_host,
            })

    else:
        raise ValueError(
            "Invalid baseline name. Choose from "
            "['characterai', 'human_simulacra', 'naive_human_simulacra', "
            "'opencharacter', 'consistent_llm', 'human_interview', "
            "'persona_hub', 'twin_2k_500', 'deeppersona', 'llm_generated']"
        )
    
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
    
    # Collect stats from all persona runs
    all_persona_stats = []
    run_start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(main, args, interviewee_kwarg): interviewee_kwarg for interviewee_kwarg in proceed_list}
        for future in as_completed(futures):
            interviewee_kwarg = futures[future]
            try:
                persona_stats = future.result()
                if persona_stats:
                    all_persona_stats.append(persona_stats)
            except Exception as e:
                logging.exception(f"Unhandled exception for interviewee {interviewee_kwarg.get('name', 'unknown')}, baseline: {interviewee_kwarg['baseline_name']}: {e}")
                # Still track failed personas
                all_persona_stats.append({
                    "name": interviewee_kwarg.get('name', 'unknown'),
                    "ai_detected": False,
                    "success": False,
                    "error_type": str(e),
                    "duration_min": 0.0,
                    "total_cost": 0.0,
                    "agents_cost": 0.0,
                    "interviewee_cost": 0.0,
                    "tool_costs": 0.0,
                    "num_interviewee_responses": 0,
                    "num_turns_completed": 0,
                    "num_tool_calls": 0,
                    "sessions_completed": 0,
                    # Failed personas get 0.0 for inter-session
                    "eval_internal_harmonic_mean": None,
                    "eval_internal_responsiveness": None,
                    "eval_internal_consistency": None,
                    "eval_external_wilson": None,
                    "eval_stability_inter_session": 0.0,
                    "eval_stability_intra_session": None,
                })
    
    # Aggregate baseline statistics
    total_personas = len(all_persona_stats)
    if total_personas > 0:
        ai_detected_count = sum(1 for s in all_persona_stats if s.get("ai_detected", False))
        success_count = sum(1 for s in all_persona_stats if s.get("success", False))
        total_duration = sum(s.get("duration_min", 0.0) for s in all_persona_stats)
        total_cost = sum(s.get("total_cost", 0.0) for s in all_persona_stats)
        total_agents_cost = sum(s.get("agents_cost", 0.0) for s in all_persona_stats)
        total_interviewee_cost = sum(s.get("interviewee_cost", 0.0) for s in all_persona_stats)
        total_tool_costs = sum(s.get("tool_costs", 0.0) for s in all_persona_stats)
        total_responses = sum(s.get("num_interviewee_responses", 0) for s in all_persona_stats)
        total_turns = sum(s.get("num_turns_completed", 0) for s in all_persona_stats)
        total_tool_calls = sum(s.get("num_tool_calls", 0) for s in all_persona_stats)
        total_sessions = sum(s.get("sessions_completed", 0) for s in all_persona_stats)
        
        baseline_summary = {
            "baseline_name": args.baseline_name,
            "run_timestamp": time.strftime('%Y-%m-%d_%H-%M-%S'),
            "run_duration_min": (time.time() - run_start_time) / 60,
            "config": {
                "questioner_model": args.questioner_model,
                "extractor_model": args.extractor_model,
                "web_search_model": args.web_search_model,
                "evaluator_model": args.evaluator_model,
                "simulator_model": args.simulator_model,
                "nhd_model": args.nhd_model,
                "num_turns": args.num_turns,
                "num_sessions": args.num_sessions,
            },
            "persona_counts": {
                "total": total_personas,
                "success": success_count,
                "failed": total_personas - success_count,
                "ai_detected": ai_detected_count,
            },
            "ai_detection_rate": ai_detected_count / total_personas,
            "success_rate": success_count / total_personas,
            "duration": {
                "total_min": total_duration,
                "avg_per_persona_min": total_duration / total_personas,
            },
            "costs": {
                "total": total_cost,
                "avg_per_persona": total_cost / total_personas,
                "breakdown": {
                    "agents_total": total_agents_cost,
                    "interviewee_total": total_interviewee_cost,
                    "tools_total": total_tool_costs,
                },
            },
            "interactions": {
                "total_interviewee_responses": total_responses,
                "avg_responses_per_persona": total_responses / total_personas,
                "total_turns_completed": total_turns,
                "avg_turns_per_persona": total_turns / total_personas,
                "total_tool_calls": total_tool_calls,
                "avg_tool_calls_per_persona": total_tool_calls / total_personas,
                "total_sessions_completed": total_sessions,
            },
            "per_persona_details": all_persona_stats,
        }
        
        # Add evaluation score aggregates if --do_eval was set
        if args.do_eval:
            # Helper function to compute average of non-None values
            def avg_score(key):
                values = [s.get(key) for s in all_persona_stats if s.get(key) is not None]
                return sum(values) / len(values) if values else None
            
            # For inter-session: include 0.0 scores from failed/AI-detected personas
            def avg_inter_session_score():
                values = [s.get("eval_stability_inter_session") for s in all_persona_stats 
                          if s.get("eval_stability_inter_session") is not None]
                return sum(values) / len(values) if values else None
            
            baseline_summary["evaluation_scores"] = {
                "internal": {
                    "avg_harmonic_mean": avg_score("eval_internal_harmonic_mean"),
                    "avg_responsiveness_score": avg_score("eval_internal_responsiveness"),
                    "avg_consistency_score": avg_score("eval_internal_consistency"),
                },
                "external": {
                    "avg_wilson_score": avg_score("eval_external_wilson"),
                },
                "stability": {
                    "avg_inter_session_score": avg_inter_session_score(),
                    "avg_intra_session_score": avg_score("eval_stability_intra_session"),
                },
            }
        
        # Save baseline summary with timestamp
        summary_path = f"{args.output_dir}/{args.baseline_name}/baseline_summary_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        write_json(baseline_summary, summary_path)
        
        # Log summary
        logging.info("="*60)
        logging.info(f"BASELINE SUMMARY: {args.baseline_name}")
        logging.info("="*60)
        logging.info(f"Total personas: {total_personas}")
        logging.info(f"Success: {success_count} ({success_count/total_personas*100:.1f}%)")
        logging.info(f"AI Detected: {ai_detected_count} ({ai_detected_count/total_personas*100:.1f}%)")
        logging.info(f"Avg duration per persona: {total_duration/total_personas:.2f} min")
        logging.info(f"Avg interviewee responses: {total_responses/total_personas:.1f}")
        logging.info(f"Total cost: ${total_cost:.4f} (avg ${total_cost/total_personas:.4f}/persona)")
        
        if args.do_eval and "evaluation_scores" in baseline_summary:
            eval_scores = baseline_summary["evaluation_scores"]
            logging.info("-"*40)
            logging.info("EVALUATION SCORES (averages):")
            internal = eval_scores.get("internal", {})
            external = eval_scores.get("external", {})
            stability = eval_scores.get("stability", {})
            logging.info(f"  Internal - Harmonic Mean: {internal.get('avg_harmonic_mean', 'N/A')}")
            logging.info(f"  Internal - Responsiveness: {internal.get('avg_responsiveness_score', 'N/A')}")
            logging.info(f"  Internal - Consistency: {internal.get('avg_consistency_score', 'N/A')}")
            logging.info(f"  External - Wilson: {external.get('avg_wilson_score', 'N/A')}")
            logging.info(f"  Stability - Inter-session: {stability.get('avg_inter_session_score', 'N/A')}")
            logging.info(f"  Stability - Intra-session: {stability.get('avg_intra_session_score', 'N/A')}")
        
        logging.info(f"Summary saved to: {summary_path}")
        logging.info("="*60)
    
    logging.info("All sessions completed.")
