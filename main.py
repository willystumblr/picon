import os
from src.utils import setup_logging, read_json, write_json, get_user_input_with_timeout
from src.env.interrogation_env import InterrogationEnv
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run the interrogation environment.")
    parser.add_argument('--baseline_name', type=str, required=True, help='Baseline name for the interviewee simulator.', choices=['characterai', 'human_simulacra', 'opencharacter', 'human_interview'])
    parser.add_argument('--model', type=str, default=None, help='Model name for the interrogation.')
    parser.add_argument('--nhd_model', type=str, default="gemini/gemini-2.5-flash", help='Model name for the NHD detector in the interviewee simulator.')
    parser.add_argument('--num_turns', type=int, default=30, help='Maximum number of turns in the interrogation.')
    parser.add_argument('--sample', action='store_true', help='Whether to sample OpenCharacter personas.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling personas.')
    parser.add_argument('--log_to_file', action='store_true', help='Whether to log to a file.')
    parser.add_argument('--questioner_prompt_path', type=str, default='src/agents/prompts/questioner.txt', help='Path to the questioner agent prompt file.')
    parser.add_argument('--entity_extractor_prompt_path', type=str, default='src/agents/prompts/entity_extractor.txt', help='Path to the entity extractor agent prompt file.')
    parser.add_argument('--claim_extractor_prompt_path', type=str, default='src/agents/prompts/claim_extractor_prompt.txt', help='Path to the claim extractor agent prompt file.')
    parser.add_argument('--web_search_prompt_path', type=str, default='src/agents/prompts/websearch_prompt.txt', help='Path to the web search agent prompt file.')
    parser.add_argument('--kg_agent_prompt_path', type=str, default='src/agents/prompts/kg_agent_prompt.txt', help='Path to the KG agent prompt file.')
    parser.add_argument('--use_claim_extractor', action='store_true', help='Whether to use the claim extractor agent instead of the entity extractor agent.')
    parser.add_argument('--output_dir', type=str, default='data/results', help='Directory to save the results.')
    parser.add_argument('--temp_output_dir', type=str, default='data/temp_results', help='Directory to save temporary results in case of errors.')
    
    return parser.parse_args()

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
                "nhd_model": args.nhd_model
            })    
    elif args.baseline_name == "human_simulacra":
        interviewee_kwargs = [{
            "baseline_name": "human_simulacra",
            "name": name,            
            "nhd_model": args.nhd_model
        } for name in ["Mary Jones", "Haley Collins", "Sara Ochoa", "James Jones", "Tami Clark", "Michael Miller", "Kevin Kelly", "Erica Walker", "Leslie Nichols", "Robert Scott", "Marsh Zhaleh"]]
    elif args.baseline_name == "opencharacter":
        dataset = load_dataset("xywang1/OpenCharacter", "Synthetic-Character", split="train")
        if args.sample:
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
                "nhd_model": args.nhd_model                
            })
    elif args.baseline_name == "human_interview":
        interviewee_kwargs = [{
            "baseline_name": "human_interview",
            "name": input("Enter your name: "),
            "nhd_model": args.nhd_model
        }]
    else:
        raise ValueError("Invalid baseline name. Choose from ['characterai', 'human_simulacra', 'opencharacter', 'human_interview']")
    
    for interviewee_kwarg in interviewee_kwargs:
        try:
            logging.info(f"Proceed to the interview session? [Y/N] (Interviewee: {interviewee_kwarg.get('name', 'unknown')}, baseline: {interviewee_kwarg['baseline_name']})")
            # proceed if y or no input for 10 seconds, else skip
            user_input = get_user_input_with_timeout(10)
            if user_input and user_input.lower() != 'y':
                logging.info(f"Skipping the interview session for interviewee: {interviewee_kwarg.get('name', 'unknown')}, baseline: {interviewee_kwarg['baseline_name']}")
                continue
            logging.info(f"Starting new session with interviewee: {interviewee_kwarg.get('name', 'unknown')}, baseline: {interviewee_kwarg['baseline_name']}")
            tools = {
                "google_claim_search": GoogleClaimSearch(
                    api_key=os.getenv('GOOGLE_CLAIM_SEARCH'),
                    cx=os.getenv('GOOGLE_CX_ID'),
                ),
                "google_geocode_validate": GoogleGeocodeValidate(api_key=os.getenv('GOOGLE_GEOCODE'))
            }
            env = InterrogationEnv(
                model=args.model,
                agents = {
                    "questioner": get_agent("questioner", args.questioner_prompt_path, model=args.model),
                    "extractor": get_agent("claim_extractor", args.claim_extractor_prompt_path, model=args.model) if args.use_claim_extractor else get_agent("entity_extractor", args.entity_extractor_prompt_path, model=args.model),
                    "web_search": get_agent("web_search", args.web_search_prompt_path, model=args.model),
                    "kg_agent": get_agent("kg_agent", args.kg_agent_prompt_path, model=args.model)
                },
                tools=tools,
                max_turns=args.num_turns,
                **interviewee_kwarg
            )
            state = env.reset()
            done = False
            while not done:
                state, done = env.step()
            state = env.finalize()
            result_path = f"{args.output_dir}/{args.baseline_name}/{interviewee_kwarg.get('name', 'unknown').replace(' ', '_')}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
            env.save_state(result_path)
        except Exception as e:
            logging.exception(f"Error during session with interviewee {interviewee_kwarg.get('name', 'unknown')}, baseline: {interviewee_kwarg['baseline_name']}: {e}")
            logging.info("Saving partial state...")
            env.save_state(f"{args.temp_output_dir}/{args.baseline_name}/{interviewee_kwarg.get('name', 'unknown').replace(' ', '_')}_error_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json", termination_status=f"Error: {str(e)}")
            continue
