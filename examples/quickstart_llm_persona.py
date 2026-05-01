"""
Quickstart: Evaluate an LLM-backed persona (no external server needed)

This pattern covers: Nemotron, Twin-2K-500, LLM-Generated, DeepPersona, and any
custom persona text. The persona is passed directly to picon.run() as a string;
picon uses the specified LLM model to role-play that persona during the interview.

Usage:
    python examples/quickstart_llm_persona.py
    python examples/quickstart_llm_persona.py --source nemotron
    python examples/quickstart_llm_persona.py --source twin
    python examples/quickstart_llm_persona.py --source llm_generated
    python examples/quickstart_llm_persona.py --source custom

Prerequisites:
    pip install picon-eval
    Set OPENAI_API_KEY or GEMINI_API_KEY in .env
"""
import argparse
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("quickstart_llm_persona")

NUM_TURNS = 5
NUM_SESSIONS = 1
INTERVIEWEE_MODEL = "gemini/gemini-2.5-flash"


def load_nemotron_persona():
    """Load one persona from nvidia/Nemotron-Personas-USA."""
    from datasets import load_dataset
    from picon.env.interviewee_simulator.persona_prompt_builders import build_nemotron_prompt

    ds = load_dataset("nvidia/Nemotron-Personas-USA", split="train")
    sample = ds[0]
    name = f"Nemotron-USA-{sample.get('uuid', '0')[:8]}"
    persona = build_nemotron_prompt(sample)
    return name, persona


def load_twin_persona():
    """Load one persona from LLM-Digital-Twin/Twin-2K-500."""
    import json
    from datasets import load_dataset
    from picon.env.interviewee_simulator.persona_prompt_builders import build_twin_2k_500_prompt

    ds = load_dataset("LLM-Digital-Twin/Twin-2K-500", "full_persona", split="data")
    sample = ds[0]
    name = f"Twin-{sample['pid']}"
    persona = build_twin_2k_500_prompt(json.dumps(sample["persona_json"]))
    return name, persona


def load_llm_generated_persona():
    """Load one descriptive persona from Tianyi-Lab/Personas."""
    from datasets import load_dataset
    from picon.env.interviewee_simulator.persona_prompt_builders import (
        build_llm_generated_prompt,
        extract_llm_generated_name,
    )

    ds = load_dataset("Tianyi-Lab/Personas", split="train")
    sample = ds[0]
    available = [c[: -len("_descriptive_persona")] for c in ds.column_names if c.endswith("_descriptive_persona")]
    prefix = next((p for p in ["Llama-3.1-70B-Instruct"] if p in available), available[0])

    raw = {
        "descriptive_persona": sample.get(f"{prefix}_descriptive_persona", ""),
        "objective_table_persona": sample.get(f"{prefix}_objective_table_persona", ""),
        "subjective_table_persona": sample.get(f"{prefix}_subjective_table_persona", ""),
        "meta_persona": sample.get("meta_persona", ""),
    }
    import json
    persona = build_llm_generated_prompt(json.dumps(raw), persona_type="descriptive")
    name = extract_llm_generated_name(raw["descriptive_persona"]) or f"LLM-Persona-{sample.get('persona_number', '0')}"
    return name, persona


def load_custom_persona():
    """A hand-written example persona — replace with your own."""
    name = "Alex"
    persona = (
        "You are Alex, a 34-year-old civil engineer based in Austin, Texas. "
        "You grew up in a mid-sized city in Ohio and moved to Texas for work after college. "
        "You are married with two children and enjoy woodworking and cycling on weekends. "
        "You are pragmatic and direct, but warm with people you trust. "
        "Respond naturally as Alex would in a casual interview."
    )
    return name, persona


SOURCES = {
    "nemotron": load_nemotron_persona,
    "twin": load_twin_persona,
    "llm_generated": load_llm_generated_persona,
    "custom": load_custom_persona,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=list(SOURCES), default="custom",
                        help="Persona source (default: custom)")
    parser.add_argument("--model", default=INTERVIEWEE_MODEL,
                        help=f"Interviewee LLM model (default: {INTERVIEWEE_MODEL})")
    parser.add_argument("--num_turns", type=int, default=NUM_TURNS)
    parser.add_argument("--num_sessions", type=int, default=NUM_SESSIONS)
    parser.add_argument("--do_eval", action="store_true", help="Run evaluation after interview")
    args = parser.parse_args()

    log.info(f"Loading persona from source: {args.source}")
    name, persona = SOURCES[args.source]()
    log.info(f"Persona name: {name}")
    log.info(f"Persona (first 200 chars): {persona[:200]}...")

    import picon

    log.info("Starting picon.run() ...")
    result = picon.run(
        model=args.model,
        persona=persona,
        name=name,
        num_turns=args.num_turns,
        num_sessions=args.num_sessions,
        do_eval=args.do_eval,
        output_dir=f"data/results/example_{args.source}",
    )

    log.info("=" * 60)
    log.info("Interview complete!")
    log.info(f"Result saved to: {result.result_path}")
    if args.do_eval and result.eval_scores:
        log.info(f"Evaluation scores: {result.eval_scores}")
    else:
        log.info("(Pass --do_eval to also run evaluation)")


if __name__ == "__main__":
    main()
