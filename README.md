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
git clone https://github.com/anonymous/picon.git
cd picon
pip install -e ".[all]"
```

&nbsp;

## Tutorial

For a hands-on walkthrough of PICON's features, see the [picon_tutorial.ipynb](picon_tutorial.ipynb) notebook.
It covers installation, running interviews, evaluation, and interpreting results with worked examples.

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
#     "ic_score": 0.82,
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

## Reproducibility

To reproduce full benchmark results, run each agent's script after setting the required API keys in `.env`:

```bash
# 1. Install
pip install -e ".[all]"

# 2. Set API keys
cp .env.example .env
# Edit .env with your keys

# 3. Run each benchmark agent
bash scripts/nemotron.sh
bash scripts/twin_2k_500.sh
bash scripts/llm_generated.sh
bash scripts/deeppersona.sh        # requires: DATASET_DIR=/path/to/deeppersona
bash scripts/human_simulacra.sh   # wrapping server auto-started per character

# OpenCharacter: serve your OpenCharacter-compatible model via vLLM first
vllm serve <your-opencharacter-model> --port 8000
VLLM_BASE=http://localhost:8000/v1 VLLM_MODEL=<your-opencharacter-model> bash scripts/opencharacter.sh

# ConsistentLLM: requires fine-tuned model served via vLLM first
vllm serve anonymous/consistent_llm_llama-8b-sft-ppo-prompt --port 8001
bash scripts/consistent_llm.sh    # SIMULATOR_PORT=8001 by default

# Character.AI (requires CAI_TOKEN in .env):
bash scripts/characterai.sh
```

All scripts write results to `data/results/` and evaluation scores to `data/evaluation/`.
By default, each script randomly samples 10 personas (`SAMPLE_N=10`, `SEED=42`).
To run all personas without sampling, set `SAMPLE_N=0`:

```bash
SAMPLE_N=0 SEED=42 bash scripts/nemotron.sh     # run all personas
```

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
| **Cooperativeness** | Fraction of turns with substantive, non-evasive responses |
| **Non-contradiction Rate** | Degree to which responses remain free of contradictions |
| **Internal Consistency (IC)** | Harmonic mean of Cooperativeness and Non-contradiction Rate |
| **Coverage** | Fraction of turns containing at least one verifiable claim |
| **Non-refutation Rate** | Per-turn rate of claims not refuted by web evidence |
| **External Consistency (EC)** | Harmonic mean of Coverage and Non-refutation Rate |
| **Retest Consistency (Inter)** | Answer stability across sessions |
| **Retest Consistency (Intra)** | Answer stability within a session |

&nbsp;

&nbsp;

## Supported Persona Agent Types

PICON includes ready-to-run scripts for eight established persona agent types.
Each script handles data loading, prompt construction, parallel execution, and evaluation in one command.

| Agent Type | Data Source | Script |
|-----------|------------|--------|
| **Human Simulacra** | 11 RAG-based characters (local) | `scripts/human_simulacra.sh` |
| **OpenCharacter** | [`xywang1/OpenCharacter`](https://huggingface.co/datasets/xywang1/OpenCharacter) (HuggingFace) | `scripts/opencharacter.sh` |
| **Character.AI** | `picon/env/personas/characterai.json` (10 characters) | `scripts/characterai.sh` |
| **Nemotron** | `nvidia/Nemotron-Personas-*` — 7 regions (HuggingFace) | `scripts/nemotron.sh` |
| **DeepPersona** | Local JSON profile files | `scripts/deeppersona.sh` |
| **Twin-2K-500** | [`LLM-Digital-Twin/Twin-2K-500`](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500) (HuggingFace) | `scripts/twin_2k_500.sh` |
| **LLM-Generated** | [`Tianyi-Lab/Personas`](https://huggingface.co/datasets/Tianyi-Lab/Personas) (HuggingFace) | `scripts/llm_generated.sh` |
| **ConsistentLLM** | `picon/env/personas/consistent_llm_personas.jsonl` (local) | `scripts/consistent_llm.sh` |

### Running evaluation scripts

All scripts share the same environment variable interface:

```bash
# Run with default settings (random sample of 10 personas)
bash scripts/nemotron.sh

# Control sample size and seed
SAMPLE_N=10 SEED=42 bash scripts/twin_2k_500.sh

# Run all personas (no sampling)
SAMPLE_N=0 bash scripts/llm_generated.sh

# Control parallelism
MAX_PARALLEL=3 bash scripts/opencharacter.sh
```

> [!NOTE]
> **Human Simulacra** automatically starts a wrapping server per character — no manual setup needed. Control the simulator model with `SIMULATOR_MODEL`.
>
> **OpenCharacter** requires a vLLM server running an OpenCharacter-compatible model. Serve your model first, then point `VLLM_BASE` and `VLLM_MODEL` to it:
> ```bash
> vllm serve <your-opencharacter-model> --port 8000
> VLLM_BASE=http://localhost:8000/v1 VLLM_MODEL=<your-opencharacter-model> bash scripts/opencharacter.sh
> ```
>
> **Character.AI** requires `CAI_TOKEN` set in your `.env` file.
>
> **DeepPersona** requires setting `DATASET_DIR` to your local data path:
> ```bash
> DATASET_DIR=/path/to/deeppersona bash scripts/deeppersona.sh
> ```
>
> **ConsistentLLM** requires a fine-tuned model served via vLLM. Use `SIMULATOR_PORT` and `SIMULATOR_MODEL` to point to the running server:
> ```bash
> vllm serve anonymous/consistent_llm_llama-8b-sft-ppo-prompt --port 8001
> bash scripts/consistent_llm.sh
> # or with a custom model:
> vllm serve <model_path> --port 8002
> SIMULATOR_PORT=8002 SIMULATOR_MODEL=hosted_vllm/<model_path> bash scripts/consistent_llm.sh
> ```
>
> **LLM-Generated** supports four persona representation styles via `PERSONA_TYPE`:
> ```bash
> PERSONA_TYPE=descriptive bash scripts/llm_generated.sh   # default
> PERSONA_TYPE=objective   bash scripts/llm_generated.sh
> PERSONA_TYPE=subjective  bash scripts/llm_generated.sh
> PERSONA_TYPE=meta        bash scripts/llm_generated.sh
> ```

&nbsp;

&nbsp;

## Citation

If you use PICON in your research, please cite:

```bibtex
@article{anonymous2026picon,
  title={PICON: A Multi-Turn Interrogation Framework for Evaluating Persona Agent Consistency},
  author={Anonymous},
  journal={arXiv preprint},
  year={2026}
}
```

&nbsp;
