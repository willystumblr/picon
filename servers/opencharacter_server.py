"""
OpenCharacter wrapping server — loads a persona from the xywang1/OpenCharacter dataset
and serves it as an OpenAI-compatible endpoint backed by a vLLM instance.

Usage:
    # 1) Start vLLM with the OpenCharacter SFT model
    vllm serve anonymous/opencharacter-sft-llama-3-8b-instruct --port 8000

    # 2) Start this wrapping server (persona baked in)
    python servers/opencharacter_server.py \\
        --port 8001 \\
        --vllm_base http://localhost:8000/v1 \\
        --vllm_model anonymous/opencharacter-sft-llama-3-8b-instruct \\
        --character_name "Jack \\"Speedy\\" Thompson"

    # 3) PICON — only the URL is needed
    picon --agent_api_base http://localhost:8001/v1 --do_eval

    # Alternatively, provide persona directly via --persona / --profile flags.
    # If --character_name is given, the dataset is searched at startup.
"""
import argparse
import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from openai import OpenAI

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from picon.env.interviewee_simulator.persona_prompt_builders import build_opencharacter_prompt

logging.basicConfig(level=logging.INFO)
app = FastAPI()

client: OpenAI = None
vllm_model: str = ""
system_prompt: str = ""
agent_name: str = ""


def generate_response(messages: list) -> str:
    if messages and messages[0]["role"] == "system":
        messages = [{"role": "system", "content": system_prompt}] + messages[1:]
    else:
        messages = [{"role": "system", "content": system_prompt}] + messages

    response = client.chat.completions.create(
        model=vllm_model,
        messages=messages,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


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
        logging.error(f"OpenCharacter error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {
        "id": f"chatcmpl-oc-{int(time.time())}",
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
    parser = argparse.ArgumentParser(description="OpenCharacter wrapping server for PICON")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--vllm_base", type=str, default="http://localhost:8000/v1",
                        help="vLLM endpoint URL")
    parser.add_argument("--vllm_model", type=str,
                        default="anonymous/opencharacter-sft-llama-3-8b-instruct")
    parser.add_argument("--character_name", type=str, default=None,
                        help="Character name to look up in xywang1/OpenCharacter dataset")
    parser.add_argument("--persona", type=str, default=None,
                        help="Persona text (overrides dataset lookup)")
    parser.add_argument("--profile", type=str, default=None,
                        help="Character profile text (used together with --persona)")
    parser.add_argument("--name", type=str, default=None,
                        help="Agent name for response metadata (defaults to --character_name)")
    args = parser.parse_args()

    if args.persona:
        system_prompt = build_opencharacter_prompt(args.persona, args.profile or "")
        agent_name = args.name or "OpenCharacter"
    elif args.character_name:
        import re
        from datasets import load_dataset
        logging.info(f"Loading xywang1/OpenCharacter dataset, searching for: {args.character_name}")
        dataset = load_dataset("xywang1/OpenCharacter", "Synthetic-Character", split="train")
        candidates = [
            d for d in dataset
            if re.match(r"Name:\s(.*)\n", d["character"]) and
               re.match(r"Name:\s(.*)\n", d["character"]).group(1).strip() == args.character_name
        ]
        if not candidates:
            raise ValueError(f"Character '{args.character_name}' not found in dataset")
        d = candidates[0]
        system_prompt = build_opencharacter_prompt(d["persona"], d["character"])
        agent_name = args.name or args.character_name
        logging.info(f"Loaded character: {agent_name}")
    else:
        raise ValueError("Provide either --character_name or --persona")

    client = OpenAI(base_url=args.vllm_base, api_key="no-key")
    logging.info(f"vLLM: {args.vllm_base} / {args.vllm_model}")
    logging.info(f"PICON endpoint: http://localhost:{args.port}/v1")
    uvicorn.run(app, host=args.host, port=args.port)
