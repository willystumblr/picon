"""
LLM-Generated Persona wrapping server — loads a persona from Tianyi-Lab/Personas
and serves it as an OpenAI-compatible endpoint.

Usage:
    python servers/llm_generated_server.py \\
        --port 8001 \\
        --model gemini/gemini-2.5-flash \\
        --persona_number 42 \\
        --persona_type descriptive

    # Or provide a persona text file directly:
    python servers/llm_generated_server.py \\
        --port 8001 \\
        --model gemini/gemini-2.5-flash \\
        --persona_file /tmp/my_persona.txt \\
        --name "LLM-Persona-42"

    # PICON — only the URL is needed
    picon --agent_api_base http://localhost:8001/v1 --do_eval
"""
import argparse
import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from picon.utils import get_completion

logging.basicConfig(level=logging.INFO)
app = FastAPI()

system_prompt: str = ""
agent_name: str = ""
model: str = ""


def generate_response(messages: list) -> str:
    if messages and messages[0]["role"] == "system":
        messages = [{"role": "system", "content": system_prompt}] + messages[1:]
    else:
        messages = [{"role": "system", "content": system_prompt}] + messages

    res = get_completion(model=model, messages=messages)
    return res.choices[0].message.content.strip()


@app.get("/")
async def root():
    return {"status": "ok", "name": agent_name}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        return JSONResponse(status_code=400, content={"error": "No messages provided"})

    try:
        content = generate_response(messages)
    except Exception as e:
        logging.error(f"LLM-Generated error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {
        "id": f"chatcmpl-llmgen-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": agent_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-Generated Persona wrapping server for PICON")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model", type=str, default="gemini/gemini-2.5-flash",
                        help="LLM model for generating responses")
    parser.add_argument("--persona_number", type=int, default=None,
                        help="persona_number field in Tianyi-Lab/Personas to load")
    parser.add_argument("--persona_type", type=str, default="descriptive",
                        choices=["descriptive", "subjective", "objective", "meta"])
    parser.add_argument("--persona_file", type=str, default=None,
                        help="Path to a .txt file containing the system prompt (overrides dataset lookup)")
    parser.add_argument("--name", type=str, default=None,
                        help="Agent name for response metadata")
    args = parser.parse_args()

    if args.persona_file:
        with open(args.persona_file) as f:
            system_prompt = f.read()
        agent_name = args.name or os.path.splitext(os.path.basename(args.persona_file))[0]
    elif args.persona_number is not None:
        from datasets import load_dataset
        from picon.env.interviewee_simulator.persona_prompt_builders import (
            build_llm_generated_prompt,
            extract_llm_generated_name,
        )
        logging.info("Loading Tianyi-Lab/Personas dataset...")
        dataset = load_dataset("Tianyi-Lab/Personas", split="train")
        matches = [d for d in dataset if d.get("persona_number") == args.persona_number]
        if not matches:
            raise ValueError(f"persona_number {args.persona_number} not found in dataset")
        data = matches[0]
        preferred_prefixes = ["Llama-3.1-70B-Instruct"]
        available_prefixes = [
            col[:-len("_descriptive_persona")]
            for col in dataset.column_names
            if col.endswith("_descriptive_persona")
        ]
        selected_prefix = next(
            (p for p in preferred_prefixes if p in available_prefixes),
            available_prefixes[0] if available_prefixes else None,
        )
        persona_data = {
            "descriptive_persona": data.get(f"{selected_prefix}_descriptive_persona", ""),
            "objective_table_persona": data.get(f"{selected_prefix}_objective_table_persona", ""),
            "subjective_table_persona": data.get(f"{selected_prefix}_subjective_table_persona", ""),
            "meta_persona": data.get("meta_persona", ""),
        }
        system_prompt = build_llm_generated_prompt(persona_data, args.persona_type)
        agent_name = args.name or f"LLM-Persona-{args.persona_number}"
        logging.info(f"Loaded persona: {agent_name}")
    else:
        raise ValueError("Provide either --persona_number or --persona_file")

    model = args.model
    logging.info(f"Model: {model}")
    logging.info(f"PICON endpoint: http://localhost:{args.port}/v1")
    uvicorn.run(app, host=args.host, port=args.port, log_level="error", access_log=False)
