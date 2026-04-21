import argparse
import asyncio
import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from PyCharacterAI import get_client

logging.basicConfig(level=logging.INFO)
app = FastAPI()

CAI_TIMEOUT = 30
client = None
chat_id = None
character_id = None 


async def setup_client(user_id: str, char_id: str):
    global client, chat_id, character_id
    character_id = char_id
    client = await asyncio.wait_for(get_client(user_id), timeout=CAI_TIMEOUT)
    _, (chat, _) = await asyncio.wait_for(
        asyncio.gather(
            client.account.fetch_me(),
            client.chat.create_chat(char_id),
        ),
        timeout=CAI_TIMEOUT,
    )
    chat_id = chat.chat_id
    logging.info(f"CharacterAI session established. chat_id={chat_id}")


@app.get("/")
async def root():
    return {"status": "ok", "chat_id": chat_id}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        return JSONResponse(status_code=400, content={"error": "No messages provided"})

    user_message = messages[-1].get("content", "")

    try:
        await asyncio.sleep(0.5) 
        response = await asyncio.wait_for(
            client.chat.send_message(
                character_id=character_id,
                chat_id=chat_id,
                text=user_message,
            ),
            timeout=CAI_TIMEOUT,
        )
        content = response.get_primary_candidate().text
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"error": "CharacterAI timeout"})
    except Exception as e:
        logging.error(f"CharacterAI error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {
        "id": f"chatcmpl-cai-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "characterai",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--character_id", type=str, required=True)
    parser.add_argument("--user_id", type=str, required=True)
    args = parser.parse_args()

    async def main():
        await setup_client(args.user_id, args.character_id)
        config = uvicorn.Config(app, host="0.0.0.0", port=args.port)
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(main())