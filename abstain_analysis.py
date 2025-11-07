from typing import List, Dict, Any, Literal
from dotenv import load_dotenv
import argparse
from src.utils import read_json, setup_logging, write_json, get_completion
from concurrent.futures import ThreadPoolExecutor
import tqdm
from pydantic import BaseModel, Field
import os
import logging

PROMPT = """You are a precise abstain classifier.

# Task
Decide whether the given answer *intentionally abstains* from answering the question.

# Definition of Abstain
An answer is **abstaining** ONLY IF:
- It clearly refuses to answer (“I cannot answer that”, “I don’t know”, “I’m not allowed to say”, etc.), OR
- It explicitly avoids the question (“That’s not relevant”, “Let’s talk about something else”), OR
- It repeats or rephrases the question without giving new information, OR
- It provides a generic or evasive statement that shows no intent to answer the question.

# NOT Abstain
An answer is **NOT abstaining** if:
- It misunderstands the question but still tries to answer it,
- It gives a wrong or irrelevant answer but in good faith,
- It answers partially, incompletely, or with minor misunderstanding,
- It speculates or guesses an answer (as long as it’s an attempt).

# Output Format
Return JSON:
{
  "abstain": true | false,
  "reason": "short explanation",
  "abstain_type": "refusal | lack info | asking back | unrelated"
}

# Example
Q: "Exact role title on SNUBH contract?"
A: "My role is to develop a Real-Time Prediction and Early Intervention Model..."
→ This is a **misunderstanding**, not abstain.
Output:
{
  "abstain": false,
  "reason": "The answer misunderstands the question but attempts to answer.",
  "abstain_type": null
}

Q: "What is your position at SNUBH?"
A: "I’m not allowed to disclose that information."
→ True abstain.
Output:
{
  "abstain": true,
  "reason": "The answer explicitly refuses to answer.",
  "abstain_type": "refusal"
}

"""
class OutputSchema(BaseModel):
    abstain: Literal['true', 'partially true', 'false'] = Field(..., description="Whether the provided response is abstaining from answering the question.")
    reason : str = Field(..., description="A brief explanation for the abstention decision.")
    abstain_type: Literal['none', 'refusal', 'lack info', 'asking back', 'unrelated'] = Field(..., description="The type of abstention (e.g., 'lack info', 'unknown', etc.).")

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
    qa_pairs = [turn['environment_observation'][0]['response'] for turn in input_data['history'] if turn['environment_observation'][0]['observation_type']=="interviewee_response" and turn['type']!="repeat"]
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
                temperature=1.0
            ), qa_pairs), total=len(qa_pairs), desc="Evaluating QA pairs")
        )

    logging.info("Evaluation completed.")
    
    outputs = [OutputSchema.model_validate_json(res.choices[0].message.content) if res.choices[0].message.content else None for res in results]
    total_abstain_count = sum(1 for res in outputs if res and res.abstain!='false')    
    print(f"Total abstentions: {total_abstain_count} out of {len(qa_pairs)}")
    for i, res in enumerate(results):
        content = res.choices[0].message.content
        parsed_content = OutputSchema.model_validate_json(content)  # Validate the response
        
        result_dict['results'].append({
            "question": qa_pairs[i]['question'],
            "answer": qa_pairs[i]['content'],
            "abstain": parsed_content.abstain,
            "reason": parsed_content.reason,
            "abstain_type": parsed_content.abstain_type
        })
    result_dict['abstain_rate'] = total_abstain_count / len(qa_pairs)
    logging.info(f"Results saved to {args.output_dir}")
    output_filename = os.path.basename(args.input_file).replace('.json', '_abstain_results.json')
    write_json(result_dict, os.path.join(args.output_dir, args.baseline_name, output_filename))

if __name__ == "__main__":
    args = parse_arguments()
    setup_logging(log_to_file=True, process_name="abstain_analysis")
    load_dotenv()
    os.makedirs(os.path.join(args.output_dir, args.baseline_name), exist_ok=True)
    abstain_eval(args)
    