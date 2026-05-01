"""
DeepPersona wrapping server — loads a profile from DeepPersona dataset files
and serves it as an OpenAI-compatible endpoint.

DeepPersona dataset files are JSON files where each key (e.g. "Profile_R1_A1")
maps to a profile dict.

Usage:
    python servers/deeppersona_server.py \\
        --port 8001 \\
        --model gemini/gemini-2.5-flash \\
        --dataset_file /path/to/profile_01.json \\
        --profile_key Profile_R1_A1

    # Or provide a persona text file directly:
    python servers/deeppersona_server.py \\
        --port 8001 \\
        --model gemini/gemini-2.5-flash \\
        --persona_file /tmp/my_persona.txt \\
        --name "Profile_R1_A1"

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
        logging.error(f"DeepPersona error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {
        "id": f"chatcmpl-dp-{int(time.time())}",
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
    parser = argparse.ArgumentParser(description="DeepPersona wrapping server for PICON")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model", type=str, default="gemini/gemini-2.5-flash",
                        help="LLM model for generating responses")
    parser.add_argument("--dataset_file", type=str, default=None,
                        help="Path to a DeepPersona profile JSON file")
    parser.add_argument("--profile_key", type=str, default=None,
                        help="Key inside the JSON file (e.g. Profile_R1_A1)")
    parser.add_argument("--persona_file", type=str, default=None,
                        help="Path to a .txt file containing the system prompt (overrides dataset lookup)")
    parser.add_argument("--name", type=str, default=None,
                        help="Agent name for response metadata")
    args = parser.parse_args()

    if args.persona_file:
        with open(args.persona_file) as f:
            system_prompt = f.read()
        agent_name = args.name or os.path.splitext(os.path.basename(args.persona_file))[0]
    elif args.dataset_file and args.profile_key:
        import json
        from picon.env.interviewee_simulator.persona_prompt_builders import build_deeppersona_prompt
        with open(args.dataset_file) as f:
            data = json.load(f)
        profile = data.get(args.profile_key)
        if profile is None:
            raise ValueError(f"Key '{args.profile_key}' not found in {args.dataset_file}")
        system_prompt = build_deeppersona_prompt(profile)
        agent_name = args.name or args.profile_key
        logging.info(f"Loaded profile: {agent_name}")
    else:
        raise ValueError("Provide either --persona_file, or both --dataset_file and --profile_key")

    model = args.model
    logging.info(f"Model: {model}")
    logging.info(f"PICON endpoint: http://localhost:{args.port}/v1")
    uvicorn.run(app, host=args.host, port=args.port)
