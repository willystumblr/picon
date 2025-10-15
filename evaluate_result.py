from itertools import combinations
import json
from src.utils import (
    get_completion,
    read_json, 
    write_json, 
    setup_logging, 
    batch_request
)
import logging
from dotenv import load_dotenv
import os
import time
from typing import List, Dict
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

CONFLICT_PROMPT = """# Current Date: {curr_date}

# Task Description
Your task is to determine contradiction in the current user input. Examine the conversation log and determine if the current user response contradicts to the earlier interaction with you.
If the user's current response is logically inconsistent with previous responses, mark `conflict`; otherwise, output `plausible`.

# Contradiction Tracking
When given a series of quesion-response pairs, for some pair k and the current pair m where k<m, the response is `conflict` only if:
- They directly negate each other. 
- They present two mutually exclusive factual claims about the same entity 
  (e.g., different birth years, conflicting event dates).
- They are logically incompatible through necessary inference 
  (e.g., being born in 2010 but graduating from college in 2009).
You should infer hidden premises only when the contradiction is unavoidable under basic logic (e.g., timelines, ages, counts). 

The series of response are `plausible` if:
- One or both responses are vague, hyperbolic, exaggerated, sarcastic, or implausible but not directly contradictory.
- The information could both be true, even if unusual or unlikely.
- The information are unrelated.
- There is a lack of knowledge (e.g., "I don't know").

In other words, if there exists some question-response pair(s) that contradicts to the current pair, output `conflict`.
Otherwise, always output `plausible`.

# Output Format
Respond with either `plausible` or `conflict` without any additional text (without backtick).
"""

CONFLICT_PAIR_PROMPT = """# Task Description
Your task is to decide whether two (question, response) pairs are in **conflict** or **plausible** with respect to each other.

# Rules
- First, look at the factual components in each response (names, numbers, dates, places, entities).
- If there are no shared or overlapping factual components, label as `plausible`.
- If there are shared components:
  - Label as `conflict` only if the claims about them cannot both be true at the same time (negations, mutually exclusive facts).
  - Otherwise, label as `plausible`.

# Clarification
- Never use external knowledge.
- Do not judge based on plausibility, exaggeration, sarcasm, or tone.
- Only use `conflict` if the responses directly clash on the same fact.

# Output Format
Respond with either `conflict` or `plausible` only, without backtick.
"""

REPEAT_PROMPT = """You will be given a single question and two corresponding answers. Determine whether the two answers are essentially the same in meaning.
If they are, output TRUE. If they are not, output FALSE.
Do not output any additional explanation or text."""

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate interrogation results.")
    parser.add_argument('--input_file', type=str, help='Path to the directory containing result JSON files.')
    parser.add_argument('--baseline_name', type=str, help='Name of the baseline', required=True, choices=['characterai', 'human_simulacra', 'opencharacter'])
    parser.add_argument('--output_dir', type=str, default='data/eval_results/', help='Directory to save evaluation results.')
    parser.add_argument('--log_to_file', action='store_true', help='Whether to log to a file.')
    parser.add_argument('--model', type=str, default='gemini/gemini-2.5-flash', help='Model to use for evaluation.')
    parser.add_argument('--batch_process', action='store_true', help='Whether to use batch processing for evaluation.')
    parser.add_argument('--sub_batch', action='store_true', help='Whether to use sub-batching in batch processing.')
    parser.add_argument('--inline', action='store_true', help='Whether to use inline processing in batch processing.')
    return parser.parse_args()

def consistency_score(data: List[Dict[str, any]], args: argparse.Namespace) -> float:
    init_conf_turn = -1
    ### 1. initial contradiction tracking ###
    convlog_format = "Question {turn}: {question}\nResponse {turn}: {response}\n"
    convlog = ""
    messages_init_conf = []
    init_conf = None
    if len(data) == 0:
        logging.warning("No data to evaluate.")
        return {"total_pairs": 0, "initial_conflict_turn": 0, "conflict_pairs": 0, "conflict_pairs_ratio": 0.0}
    messages_init_conf.append({
        "role": "system",
        "content": CONFLICT_PROMPT.format(curr_date=data[0][1]['content'])
    })
    for i, (idx, qa_pair) in tqdm(enumerate(data[1:101]), total=len(data[1:101]), desc="Initial Conflict Detection"):
        convlog += convlog_format.format(turn=idx, question=qa_pair['question'], response=qa_pair['content'])
        messages_init_conf.append({
            "role": "user",
            "content": convlog
        })
        while True:
            res = get_completion(
                model = args.model,
                messages=messages_init_conf,
                temperature=0.0 if args.model.startswith("gemini") else 1.0
            )
            if res is not None and res.choices and res.choices[0].message and res.choices[0].message.content and res.choices[0].message.content.strip() in ['plausible', 'conflict']:
                break
        if res.choices[0].message.content.strip() == 'conflict':
            logging.info(f"Found initial conflict at turn {idx}: {qa_pair}")
            init_conf_turn = i
            init_conf = qa_pair
            break
            
    ### 2. conflict pairwise comparison ###
    filtered_data = data[1:101] # hard-coded for now
    combination = list(combinations(filtered_data, 2))
    if init_conf_turn == -1:
        logging.info("No initial conflict found.")
        return {"total_pairs": len(combination), "initial_conflict_turn": -1, "conflict_pairs": 0, "conflict_pairs_ratio": 0.0}
    
    conflict_count = 0
    pair_format = "Question 1: {question_1}\nResponse 1: {response_1}\n\nQuestion 2: {question_2}\nResponse 2: {response_2}"
    content_list = [pair_format.format(
        question_1=qa1['question'], response_1=qa1['content'],
        question_2=qa2['question'], response_2=qa2['content']
    ) for (_, qa1), (_, qa2) in combination]
    messages_list = [[
        {
            "role": "system",
            "content": CONFLICT_PAIR_PROMPT
        },
        {
            "role": "user",
            "content": content
        }
    ] for content in content_list]
    conflict_pairs = []
    if args.batch_process:
        logging.info("Using batch processing for conflict pair evaluation.")
        logging.info(f"Total pairs to evaluate: {len(messages_list)}")
        logging.info(f"Using model: {args.model}")
        logging.info(f"Sub-batch mode: {args.sub_batch}")
        logging.info("This may take a while...")
        display_name = f"consistency_eval_{int(time.time())}"
        results = batch_request(messages=messages_list, model=args.model, display_name=display_name, sub_batch=args.sub_batch, inline=args.inline)
        for i, text in enumerate(results):
            if not text or text not in ["plausible", "conflict"]:
                logging.warning(f"Unexpected response: {text}")
            if text == "conflict":
                conflict_count += 1
                conflict_pairs.append({
                    "pair_index": i,
                    "pair": combination[i]
                })
    else:
        logging.info("Using sequential processing for conflict pair evaluation.")
        logging.info(f"Total pairs to evaluate: {len(messages_list)}")
        logging.info(f"Using model: {args.model}")
        logging.info("This may take a while...")
        with ThreadPoolExecutor(max_workers=32) as executor: # using tqdm for progress bar
            results = list(tqdm(executor.map(lambda msg: get_completion(model=args.model, messages=msg, temperature=0.0), messages_list), total=len(messages_list), desc="Evaluating Conflict Pairs"))
        for i, res in enumerate(results):
            if not res or not res.choices or not res.choices[0].message or not res.choices[0].message.content or res.choices[0].message.content.strip() not in ['plausible', 'conflict']:
                logging.warning(f"Unexpected response: {res}")
                continue
            if res.choices[0].message.content.strip() == "conflict":
                conflict_count += 1
                conflict_pairs.append({
                    "pair_index": i,
                    "pair": combination[i]
                })
                
    logging.info(f"Total conflict pairs: {conflict_count} out of {len(combination)}")
    total_results = {
        "total_pairs": len(combination),
        "initial_conflict_turn": {
            "turn_index": init_conf_turn,
            "qa_pair": init_conf
        },
        "conflict_pairs": {
            "count": conflict_count,
            "pairs": conflict_pairs
        },
        "conflict_pairs_ratio": conflict_count / len(combination)
    }
    return total_results

def extract_qa_pairs(data: List[Dict[str, any]]):
    qa_data = []
    for i, turn in enumerate(data['history']):
        qa_pair = turn['environment_observation'][-1]['response']
        """{"question": "...", "content": "..."}"""
        qa_data.append((i, qa_pair))
    return qa_data

def compute_repeat_score(data: List[Dict[str, any]]) -> float:
    original_main_qa = [turn['environment_observation'][-1]['response'] for turn in data['history'] if turn['type'] == 'get_to_know']
    original_main_qa.pop(0)
    repeated_main_qa = [turn['environment_observation'][-1]['response'] for turn in data['history'] if turn['type'] == 'repeat']
    if len(original_main_qa) == 0:
        logging.warning("No original main Q&A found.")
        return 0.0
    if len(repeated_main_qa) == 0:
        logging.warning("No repeated main Q&A found.")
        return 0.0
    if len(repeated_main_qa) < len(original_main_qa):
        logging.warning("Fewer repeated main Q&A than original main Q&A. Truncating...")
        original_main_qa = original_main_qa[:len(repeated_main_qa)]

    result = []
    true_num = 0
    for i, (ori, rep) in tqdm(enumerate(zip(original_main_qa, repeated_main_qa)), total=len(original_main_qa), desc="Repeat Score Calculation"):
        # assert ori['question'] == rep['question'], "Mismatched questions in original and repeated Q&A."
        while True:
            res = get_completion(
                model="gemini/gemini-2.5-flash",
                messages=[
                    {"role": "system", "content": REPEAT_PROMPT},
                    {"role": "user", "content": f"Question: {ori['question']}\n\nResponse 1: {ori['content']}\nResponse 2: {rep['content']}"}
                ],
                temperature=0.0 if args.model.startswith("gemini") else 1.0
            )
            judge = res.choices[0].message.content.strip() if res and res.choices and res.choices[0].message and res.choices[0].message.content else "FALSE"
            if judge in ["TRUE", "FALSE"]:
                break
            logging.warning(f"Unexpected response for repeat score: {judge}. Retrying...")
        result.append({
            "question": ori['question'],
            "original_response": ori['content'],
            "repeated_response": rep['content'],
            "is_repeat": judge
        })
        true_num += (judge=='TRUE')
    result = {
        'repeat_score' : round(true_num / len(original_main_qa), 4),
        'repeat_results' : result
    }
    return result

if __name__ == "__main__":
    args = parse_args()
    load_dotenv()
    setup_logging(log_to_file=args.log_to_file, process_name="evaluate_result")
    # Example usage
    # result_path = os.path.dirname(args.result_file)
    data = read_json(args.input_file)
    qa_data = extract_qa_pairs(data)
    scores = consistency_score(qa_data, args)
    scores['repeat_score'] = compute_repeat_score(data)
    logging.info(f"Consistency Scores: {json.dumps(scores, indent=2)}")
    os.makedirs(os.path.join(args.output_dir, args.baseline_name), exist_ok=True)
    output_path = os.path.join(args.output_dir, args.baseline_name, f"eval_{os.path.basename(args.input_file)}")
    write_json(scores, output_path)
    logging.info(f"Saved evaluation results to {output_path}")
    