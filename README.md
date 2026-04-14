# PICON — Persona Interrogation framework for Consistency evaluation

---
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/picon)
![PyPI Version](https://img.shields.io/pypi/v/picon)
---

An official Python package for evaluating LLM-based persona agents, called `PICON`.
By running a multi-turn interview and fact-checking pipeline, you can measure how consistently and accurately a persona agent behaves.
PICON evaluates persona agents across three dimensions:
* **Internal Consistency**: freedom from self-contradiction across answers.
* **External Consistency**: alignment of claims with real-world facts (via web search).
* **Retest Stability**: consistency of answers when the same questions are repeated within and across sessions.

&nbsp;

### Recent updates
* *April 2026 (v0.1.3)*: AWS Bedrock Claude compatibility — `reasoning_effort`/`thinking` are automatically stripped for all Claude-family models (including `bedrock/anthropic.claude-*` and `anthropic/claude-*`). Reasoning/thinking is also disabled by default for interviewee API calls to keep persona replies direct.
* *March 2026 (v0.1.0)*: Initial release with interview pipeline, evaluation, and CLI.

&nbsp;

&nbsp;

## Installation

```bash
pip install picon-eval
```
```python
import picon
print(picon.__version__)
```

For development or full extras (CharacterAI, Google GenAI, etc.):
```bash
git clone https://github.com/willystumblr/picon.git
cd picon
pip install -e ".[all]"
```

&nbsp;

&nbsp;

## Quick Starts

> [!NOTE]
> Before using PICON, you must provide API keys either directly or in a `.env` file.
> * *OpenAI models (gpt-\*)*: Set `OPENAI_API_KEY` in your `.env` file.
> * *Gemini models (gemini/\*)*: Set `GEMINI_API_KEY` in your `.env` file.
> * *Web search (external verification)*: Set `SERPER_API_KEY` in your `.env` file. Get one at [serper.dev](https://serper.dev).

&nbsp;

### Environment Variables

Create a `.env` file in your working directory:

```bash
# LLM API Keys (at least one required)
OPENAI_API_KEY="YOUR_OPENAI_KEY"
GEMINI_API_KEY="YOUR_GEMINI_KEY"

# Web Search (required for external verification)
SERPER_API_KEY="YOUR_SERPER_KEY"

# Address validation (required for external verification)
GOOGLE_GEOCODE="YOUR_GOOGLE_GEOCODE_KEY"

# Optional
ANTHROPIC_API_KEY="YOUR_ANTHROPIC_KEY"
GOOGLE_CLAIM_SEARCH="YOUR_GOOGLE_API_KEY"       # Fact-check search
GOOGLE_CX_ID="YOUR_CUSTOM_SEARCH_ENGINE_ID"     # Custom Search Engine ID

# AWS Bedrock (for bedrock/anthropic.claude-* models)
AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY"
AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_KEY"
AWS_REGION="us-west-2"

# Google Vertex AI (for vertex_ai/claude-* models)
GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

> [!TIP]
> PICON supports Claude via direct Anthropic API (`claude-*`), AWS Bedrock (`bedrock/anthropic.claude-*`), and Google Vertex AI (`vertex_ai/claude-*`). Provider-specific reasoning/thinking parameters are handled automatically.

&nbsp;

### Component-Based Usage

Import individual components and compose your own simulation pipeline:

```python
from picon import Questioner, EntityExtractor, Evaluator, Interviewee
from picon import InterrogationSimulation

# Set up agents
questioner = Questioner(model="gpt-5")
extractor = EntityExtractor(model="gpt-5.1")
evaluator = Evaluator(model="gemini/gemini-2.5-flash")

# Set up the persona to evaluate
interviewee = Interviewee(
    model="gpt-5",
    persona="You are a 35-year-old software engineer living in Seoul.",
    name="John",
)

# Run the full interview + evaluation pipeline
sim = InterrogationSimulation(
    interviewee=interviewee,
    questioner=questioner,
    extractor=extractor,
    evaluator=evaluator,
    num_turns=20,
    num_sessions=2,
)
result = sim.run(do_eval=True)

print(result.eval_scores)
result.save("results/john.json")

# Example output:
# {
#     "internal_harmonic_mean": 0.82,
#     "external_ec": 0.75,
#     "inter_session_stability": 0.68,
#     "intra_session_stability": 0.91,
# }
```

With default agent models, you can omit agent setup entirely:

```python
from picon import Interviewee, InterrogationSimulation

interviewee = Interviewee(model="gpt-5", persona="You are ...", name="John")
result = InterrogationSimulation(interviewee=interviewee, num_turns=20).run()
```

&nbsp;

### Evaluate an LLM Persona (Simple API)

For quick evaluations, use the `picon.run()` shortcut:

```python
import picon

result = picon.run(
    model="gpt-5",
    persona="You are a 35-year-old software engineer living in Seoul.",
    name="John",
    num_turns=20,
    num_sessions=2,
    do_eval=True,
)

print(result.eval_scores)
result.save("results/john.json")
```

```bash
# Equivalent CLI command
picon --agent_model gpt-5 \
      --agent_persona "You are a 35-year-old software engineer living in Seoul." \
      --agent_name "John" \
      --num_turns 20 --num_sessions 2 --do_eval
```

&nbsp;

### Evaluate an External Agent Endpoint

If you already have a persona agent running (e.g. a wrapping server, fine-tuned model, RAG agent), provide its OpenAI-compatible endpoint URL (`/v1/chat/completions`).

```python
from picon import Interviewee, InterrogationSimulation

interviewee = Interviewee(api_base="http://localhost:8000/v1", name="MyAgent")
result = InterrogationSimulation(interviewee=interviewee, num_turns=20).run()
```

```bash
# Equivalent CLI command
picon --agent_api_base http://localhost:8000/v1 \
      --agent_name "MyAgent" \
      --num_turns 20 --num_sessions 2 --do_eval
```

&nbsp;

### Self-hosted Models (vLLM)

For self-hosted models, provide both `api_base` and `model`:

```python
from picon import Interviewee, InterrogationSimulation

interviewee = Interviewee(
    api_base="http://localhost:8000/v1",
    model="meta-llama/Llama-3-8B",
    persona="You are a 30-year-old teacher named Jane...",
    name="Jane",
)
result = InterrogationSimulation(interviewee=interviewee).run()
```

```bash
picon --agent_api_base http://localhost:8000/v1 \
      --agent_model meta-llama/Llama-3-8B \
      --agent_persona "You are a 30-year-old teacher named Jane..." \
      --agent_name "Jane" --do_eval
```

&nbsp;

### Separate Interview and Evaluation

```python
import picon

# Step 1: Interview only
interview_result = picon.run_interview(
    name="John",
    model="gpt-5",
    persona="You are a 35-year-old software engineer...",
    num_turns=20,
    num_sessions=2,
)

# Step 2: Evaluate
persona_stats = picon.run_evaluation(interview_result, eval_factors=["internal", "external"])
print(persona_stats)
```

### Evaluate an Existing Result File

```python
scores = picon.evaluate("results/john.json", eval_factors=["internal", "external"])
```

&nbsp;

&nbsp;

## Connecting an External Agent

PICON can evaluate any persona agent that exposes an **OpenAI-compatible** chat completions endpoint (`POST /v1/chat/completions`).
If your agent already serves this endpoint (e.g. vLLM or any OpenAI-compatible server), just pass the URL directly — no wrapping needed.

&nbsp;

### Case 1: Your agent already has an OpenAI-compatible endpoint

If you're serving a model via vLLM or any server that implements `/v1/chat/completions`:

```python
import picon

result = picon.run(
    api_base="http://<your-server-ip>:8000/v1",
    name="Alice",
    do_eval=True,
)
```

```bash
picon --agent_api_base http://<your-server-ip>:8000/v1 \
      --agent_name "Alice" --do_eval
```

&nbsp;

### Case 2: Your agent has custom logic (RAG, API calls, etc.)

If your agent doesn't have an OpenAI-compatible endpoint, wrap it with a simple server.
You only need to implement **one endpoint** that accepts `messages` and returns a response:

```python
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

def generate_response(messages: list) -> str:
    """Replace this with your own agent logic."""
    user_message = messages[-1]["content"]
    # ... your custom logic (RAG retrieval, API call, etc.) ...
    return "This is my response."

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    content = generate_response(messages)

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "my-agent",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

Then evaluate with PICON:

```bash
picon --agent_api_base http://<your-server-ip>:8001/v1 \
      --agent_name "MyAgent" --do_eval
```

See [`examples/`](examples/) for full end-to-end scripts with vLLM + LoRA and HumanSimulacra RAG

&nbsp;

&nbsp;

## How It Works

```
1. Get-to-Know        Ask predefined demographic questions (WVS dataset)
       |
2. Main Interrogation Each turn runs this agent chain:
       |
       |-- Questioner    Generate the next question based on conversation history
       |-- Interviewee   The persona under evaluation answers the question
       |-- Extractor     Pull out entities and verifiable claims from the answer
       |-- Web Search    Fact-check extracted claims against the web
       '-- Evaluator     Compare this answer with previous answers for consistency
       |
3. Repeat Phase        Re-ask the get-to-know questions to measure stability
       |
4. Finalize            Compute all evaluation scores and save results
```

&nbsp;

&nbsp;

## API Reference

### Component Classes

> `Interviewee(model, persona, name, api_base, api_key)`:
> * `model` (str): LLM model name. Required if `api_base` is not provided.
> * `persona` (str): System prompt or path to a `.txt` file. Default: `""`.
> * `name` (str): Interviewee display name. Default: `"Agent"`.
> * `api_base` (str): OpenAI-compatible endpoint URL. Required if `model` is not provided.
> * `api_key` (str): API key for the endpoint. Default: `None`.

> `Questioner(model, prompt_path)` / `EntityExtractor(model, prompt_path)` / `Evaluator(model, prompt_path)` / `WebSearch(model, prompt_path)`:
> * `model` (str): LLM model name. Each agent has its own default (see below).
> * `prompt_path` (str): Custom system prompt file. `None` uses the built-in prompt.

> `InterrogationSimulation(interviewee, questioner, extractor, web_search, evaluator, ...)`:
> * `interviewee` (Interviewee): The persona agent to evaluate. **Required.**
> * `questioner` (Questioner): Questioner agent. `None` creates one with default model.
> * `extractor` (EntityExtractor): Extractor agent. `None` creates one with default model.
> * `web_search` (WebSearch): Web search agent. `None` creates one with default model.
> * `evaluator` (Evaluator): Evaluator agent. `None` creates one with default model.
> * `num_turns` (int): Interview turns per session. Default: `30`.
> * `num_sessions` (int): Number of repeated sessions. Default: `2`.
> * `nhd_model` (str): Model for AI detection. Default: `"gpt-5-nano"`.
> * `output_dir` (str): Output directory. Default: `"data/results"`.
> * `question_seed` (int): Random seed for question selection. Default: `42`.

&nbsp;

### Simple API

> `picon.run()` / `picon.run_interview()` Parameters:
> * `name` (str): Interviewee name. Default: `"Agent"`.
> * `model` (str): LLM model name (e.g. `"gpt-5"`, `"gemini/gemini-2.5-flash"`). Required if `api_base` is not provided.
> * `persona` (str): System prompt or path to a `.txt` file. Default: `""`.
> * `api_base` (str): OpenAI-compatible API endpoint URL. Required if `model` is not provided.
> * `api_key` (str): API key for the persona endpoint. Default: `None`.
> * `num_turns` (int): Number of interview turns. Default: `30`.
> * `num_sessions` (int): Number of repeated sessions. Default: `2`.
> * `do_eval` (bool): Run evaluation after interview. Default: `True`.
> * `eval_factors` (list): Evaluation factors to run: `"internal"`, `"external"`, `"intra"`, `"inter"`. Default: `None` (all).
> * `questioner_model` (str): Model for the questioner agent. Default: `"gpt-5"`.
> * `extractor_model` (str): Model for the entity extractor agent. Default: `"gpt-5.1"`.
> * `web_search_model` (str): Model for the web search agent. Default: `"gpt-5"`.
> * `evaluator_model` (str): Model for the evaluator agent. Default: `"gemini/gemini-2.5-flash"`.
> * `nhd_model` (str): Model for AI detection. Default: `"gpt-5-nano"`.
> * `output_dir` (str): Output directory for results. Default: `"data/results"`.
> * `question_seed` (int): Random seed for question selection. Default: `42`.

&nbsp;

&nbsp;

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Internal Responsiveness** | Relevance of answers to questions |
| **Internal Consistency** | Consistency of answers to repeated questions |
| **Internal Harmonic Mean** | Harmonic mean of Responsiveness and Consistency |
| **External Coverage** | Fraction of turns containing at least one verifiable claim |
| **External Non-refutation Rate** | Per-turn rate of claims not refuted by web evidence |
| **External Consistency (EC)** | Harmonic mean of Coverage and Non-refutation Rate |
| **Inter-session Stability** | Answer stability across sessions |
| **Intra-session Stability** | Answer stability within a session |

&nbsp;

&nbsp;

## Examples

End-to-end scripts in [`examples/`](examples/):

```bash
# OpenCharacter (vLLM + LoRA)
python examples/test_opencharacter_vllm.py

# HumanSimulacra (RAG agent)
python examples/test_human_simulacra.py
python examples/test_human_simulacra.py --character "Kevin Kelly" --model "gpt-5"
```

&nbsp;

&nbsp;

## Citation

If you use PICON in your research, please cite:

```bibtex
@article{kim2026picon,
  title={PICON: A Multi-Turn Interrogation Framework for Evaluating Persona Agent Consistency},
  author={Kim, Minseo and Im, Sujeong and Choi, Junseong and Lee, Junhee and Shim, Chaeeun and Choi, Edward},
  journal={arXiv preprint arXiv:2603.25620},
  year={2026}
}
```

&nbsp;
