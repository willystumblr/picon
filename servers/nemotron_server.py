"""
Nemotron wrapping server — loads a persona from nvidia/Nemotron-Personas-* datasets
and serves it as an OpenAI-compatible endpoint.

Supported regions: usa, kor, sgp, fra, ind, jpn, bra

Usage:
    python servers/nemotron_server.py \\
        --port 8001 \\
        --model gemini/gemini-2.5-flash \\
        --region usa \\
        --uuid <uuid-from-dataset>

    # Or provide a persona text file directly:
    python servers/nemotron_server.py \\
        --port 8001 \\
        --model gemini/gemini-2.5-flash \\
        --persona_file /tmp/my_persona.txt \\
        --name "Nemotron-USA-abc123"

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

REGION_TO_DATASET = {
    "usa": ("nvidia/Nemotron-Personas-USA",       "train"),
    "kor": ("nvidia/Nemotron-Personas-Korea",     "train"),
    "sgp": ("nvidia/Nemotron-Personas-Singapore", "train"),
    "fra": ("nvidia/Nemotron-Personas-France",    "train"),
    "ind": ("nvidia/Nemotron-Personas-India",     "en_IN"),
    "jpn": ("nvidia/Nemotron-Personas-Japan",     "train"),
    "bra": ("nvidia/Nemotron-Personas-Brazil",    "train"),
}


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
        logging.error(f"Nemotron error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {
        "id": f"chatcmpl-nem-{int(time.time())}",
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
    parser = argparse.ArgumentParser(description="Nemotron Persona wrapping server for PICON")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model", type=str, default="gemini/gemini-3-flash-preview",
                        help="LLM model for generating responses")
    parser.add_argument("--region", type=str, default=None,
                        choices=list(REGION_TO_DATASET.keys()),
                        help="Nemotron region dataset to load from")
    parser.add_argument("--uuid", type=str, default=None,
                        help="UUID of the persona in the Nemotron dataset")
    parser.add_argument("--persona_file", type=str, default=None,
                        help="Path to a .txt file containing the system prompt (overrides dataset lookup)")
    parser.add_argument("--name", type=str, default=None,
                        help="Agent name for response metadata")
    args = parser.parse_args()

    if args.persona_file:
        with open(args.persona_file) as f:
            system_prompt = f.read()
        agent_name = args.name or os.path.splitext(os.path.basename(args.persona_file))[0]
    elif args.region and args.uuid:
        from datasets import load_dataset
        from picon.env.interviewee_simulator.persona_prompt_builders import build_nemotron_prompt
        repo, split = REGION_TO_DATASET[args.region]
        logging.info(f"Loading {repo} (split={split})...")
        dataset = load_dataset(repo, split=split)
        matches = [d for d in dataset if d.get("uuid") == args.uuid]
        if not matches:
            raise ValueError(f"UUID '{args.uuid}' not found in {repo}")
        data = dict(matches[0])
        system_prompt = build_nemotron_prompt(data)
        agent_name = args.name or f"Nemotron-{args.region.upper()}-{args.uuid}"
        logging.info(f"Loaded persona: {agent_name}")
    else:
        raise ValueError("Provide either --persona_file, or both --region and --uuid")

    model = args.model
    logging.info(f"Model: {model}")
    logging.info(f"PICON endpoint: http://localhost:{args.port}/v1")
    uvicorn.run(app, host=args.host, port=args.port, log_level="error", access_log=False)
