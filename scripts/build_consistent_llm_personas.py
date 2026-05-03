"""
Convert ConsistentLLM's config_chatting_personas.json → consistent_llm_personas.jsonl

Usage:
    python scripts/build_consistent_llm_personas.py \
        --personas_file <path>/chatting/config_chatting_personas.json \
        --config_file  <path>/chatting/config_chatting.json \
        --output_file  picon/env/personas/consistent_llm_personas.jsonl \
        --seed 42

Each persona is paired with a randomly chosen counterpart from the same list.
Output fields: persona, name, counterpart_name, instruction
"""

import argparse
import json
import random


INSTRUCTION_TEMPLATE = (
    "\nContinue the conversation with {counterpart}. Remember you are {name}."
    "Keep your response very brief — 2 sentences or less. "
    "Do NOT repeat anything you've already said: \n"
    'DO NOT PREFACE THE RESPONSE WITH THIRD-PERSON STATEMENTS SUCH AS '
    '"Sure, here\'s a response from..."\n'
    "{name}:"
)


def build_persona_prompt(agent_prompt_template: str, name: str, counterpart: str, backstory: str) -> str:
    return (
        agent_prompt_template
        .replace("%SPEAKER_ROLE%", name)
        .replace("%LISTENER_ROLE%", counterpart)
        .replace("%SPEAKER_BACKSTORY%", backstory)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--personas_file", default="chatting/config_chatting_personas.json")
    parser.add_argument("--config_file",   default="chatting/config_chatting.json")
    parser.add_argument("--output_file",   default="picon/env/personas/consistent_llm_personas.jsonl")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    with open(args.personas_file) as f:
        personas = json.load(f)

    with open(args.config_file) as f:
        config = json.load(f)

    agent_prompt = config["agent1_prompt"]

    rng = random.Random(args.seed)
    names = [p["name"] for p in personas]

    records = []
    for p in personas:
        name = p["name"]
        backstory = p["persona"]

        # pick a different persona as counterpart
        counterpart = rng.choice([n for n in names if n != name])

        persona_prompt = build_persona_prompt(agent_prompt, name, counterpart, backstory)
        instruction = INSTRUCTION_TEMPLATE.format(counterpart=counterpart, name=name)

        records.append({
            "persona":          persona_prompt,
            "name":             name,
            "counterpart_name": counterpart,
            "instruction":      instruction,
        })

    with open(args.output_file, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} personas to {args.output_file}")


if __name__ == "__main__":
    main()
