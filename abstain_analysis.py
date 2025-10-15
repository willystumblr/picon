from typing import List, Dict, Any
from dotenv import load_dotenv
import argparse
from src.utils import read_json, setup_logging, write_json, get_completion
from concurrent.futures import ThreadPoolExecutor
import tqdm
from pydantic import BaseModel, Field
import os
import logging

PROMPT = """You are a classifier that looks at a QA pair and determines whether the answer evades the question.

## Core Definition
- Evasion: The answer does not provide any relevant information to the question.
- Relevant information:
- Self-related: facts, opinions, preferences, experiences, plans, or personal states (first-person information).
- Topic-related: information, explanations, or statements that directly address the subject matter of the question.
- If the answer provides even partial self-related or topic-related information, it is not evasion.


## Classification Rules
1.	Check what the question is asking for:
    - If it asks for self-related information, look for self-info in the answer.
    - If it asks about a topic, look for topic-related info in the answer.
2.	If the answer contains relevant information (self-info or topic-related) → {"abstain": false, "abstain_type": "none"}
3.	If the answer contains no relevant information, classify as evasion (abstain=true) and assign one type:
    - "asking back" → A counter-question or deflection (e.g., “Why do you ask?”).
    - "lack info" → Expressing uncertainty or ignorance (e.g., “I don’t know.”).
    - "refusal" → Explicit refusal (e.g., “I don’t want to answer that.”).
    - "unrelated" → Off-topic, jokes, ads, or irrelevant talk with no relevant content, including implicit refusal (e.g., descriptive answers to Y/N question without Y/N) or slightly changing the topic.


## Priority (if multiple types apply)
- Explicit refusal → "refusal"
- Uncertainty/ignorance → "lack info"
- Counter-question → "asking back"
- Otherwise (irrelevant/off-topic) → "unrelated"


## Edge Cases
- Short but relevant info = not evasion (e.g., “Seoul.”).
- Vague but relevant info = not evasion (e.g., “Maybe tomorrow.”).
- General statements unrelated to question = evasion.
- Jokes/metaphors with no relevant info = "unrelated".


## Output Format (must match exactly)

{"abstain": <true|false>, "abstain_type": "<refusal|lack info|asking back|unrelated|none>"}
"""
class OutputSchema(BaseModel):
    abstain: bool = Field(..., description="Whether the provided response is abstaining from answering the question.")
    abstain_type: str = Field(..., description="The type of abstention (e.g., 'lack info', 'unknown', etc.).")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file containing data to evaluate.")
    parser.add_argument("--output_dir", type=str, default="data/abstain_results/", help="Path to the output file to save evaluation results.")
    parser.add_argument("--baseline_name", type=str, required=True, choices=["characterai", "human_simulacra", "opencharacter", "human_interview"], help="Baseline name for the interviewee simulator.")
    parser.add_argument("--model", type=str, default="gemini/gemini-2.5-flash", help="LLM model to use for evaluation.")

    return parser.parse_args()

def abstain_eval(args: argparse.Namespace):
    logging.info("Loading input data...")
    input_data = read_json(args.input_file)
    qa_pairs = [turn['environment_observation'][-1]['response'] for turn in input_data['history'] if turn['environment_observation'][-1]['observation_type']=="interviewee_response" and turn['type']!="repeat"]
    result_dict = {
        "model": args.model,
        "results":[]
    }
    logging.info(f"Evaluating {len(qa_pairs)} QA pairs...")
    with ThreadPoolExecutor(max_workers=16) as executor: # tqdm progress bar
        results = list(tqdm.tqdm(executor.map(
            lambda qa: get_completion(
                model=args.model,
                messages=[
                    {
                        "role": "system",
                        "content": PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"Question: {qa['question']}\nAnswer: {qa['content']}"
                    }
                ],
                response_format=OutputSchema,
                temperature=0.0
            ), qa_pairs), total=len(qa_pairs), desc="Evaluating QA pairs")
        )

    logging.info("Evaluation completed.")
    total_abstain_count = sum(1 for res in results if res.choices[0].message.content and OutputSchema.model_validate_json(res.choices[0].message.content).abstain)    
    print(f"Total abstentions: {total_abstain_count} out of {len(qa_pairs)}")
    for i, res in enumerate(results):
        content = res.choices[0].message.content
        parsed_content = OutputSchema.model_validate_json(content)  # Validate the response
        
        result_dict['results'].append({
            "question": qa_pairs[i]['question'],
            "answer": qa_pairs[i]['content'],
            "abstain": parsed_content.abstain,
            "abstain_type": parsed_content.abstain_type
        })
    logging.info(f"Results saved to {args.output_dir}")
    output_filename = os.path.basename(args.input_file).replace('.json', '_abstain_results.json')
    write_json(result_dict, os.path.join(args.output_dir, args.baseline_name, output_filename))

if __name__ == "__main__":
    args = parse_arguments()
    setup_logging(log_to_file=True, process_name="abstain_analysis")
    load_dotenv()
    os.makedirs(os.path.join(args.output_dir, args.baseline_name), exist_ok=True)
    abstain_eval(args)
    