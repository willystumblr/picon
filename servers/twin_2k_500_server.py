"""
Twin-2K-500 wrapping server — loads a persona from LLM-Digital-Twin/Twin-2K-500
and serves it as an OpenAI-compatible endpoint.

Usage:
    python servers/twin_2k_500_server.py \\
        --port 8001 \\
        --model gemini/gemini-2.5-flash \\
        --pid 123

    # Or provide a persona text file directly:
    python servers/twin_2k_500_server.py \\
        --port 8001 \\
        --model gemini/gemini-2.5-flash \\
        --persona_file /tmp/my_persona.txt \\
        --name "Twin-123"

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
        logging.error(f"Twin-2K-500 error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {
        "id": f"chatcmpl-twin-{int(time.time())}",
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
    parser = argparse.ArgumentParser(description="Twin-2K-500 wrapping server for PICON")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model", type=str, default="gemini/gemini-3-flash-preview",
                        help="LLM model for generating responses")
    parser.add_argument("--pid", type=str, default=None,
                        help="pid field in LLM-Digital-Twin/Twin-2K-500 to load")
    parser.add_argument("--persona_file", type=str, default=None,
                        help="Path to a .txt file containing the system prompt (overrides dataset lookup)")
    parser.add_argument("--name", type=str, default=None,
                        help="Agent name for response metadata")
    args = parser.parse_args()

    if args.persona_file:
        with open(args.persona_file) as f:
            system_prompt = f.read()
        agent_name = args.name or os.path.splitext(os.path.basename(args.persona_file))[0]
    elif args.pid is not None:
        import json
        from datasets import load_dataset
        from picon.env.interviewee_simulator.persona_prompt_builders import build_twin_2k_500_prompt
        logging.info("Loading LLM-Digital-Twin/Twin-2K-500 dataset...")
        dataset = load_dataset("LLM-Digital-Twin/Twin-2K-500", "full_persona", split="data")
        matches = [d for d in dataset if str(d.get("pid")) == str(args.pid)]
        if not matches:
            raise ValueError(f"pid '{args.pid}' not found in dataset")
        data = matches[0]
        system_prompt = build_twin_2k_500_prompt(json.dumps(data["persona_json"], ensure_ascii=False))
        agent_name = args.name or f"Twin-{args.pid}"
        logging.info(f"Loaded persona: {agent_name}")
    else:
        raise ValueError("Provide either --pid or --persona_file")

    model = args.model
    logging.info(f"Model: {model}")
    logging.info(f"PICON endpoint: http://localhost:{args.port}/v1")
    uvicorn.run(app, host=args.host, port=args.port, log_level="error", access_log=False)
