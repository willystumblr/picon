#!/bin/bash
# Run all 8 persona agent baselines sequentially.
#
# Prerequisites:
#   - Set API keys in .env (OPENAI_API_KEY, GEMINI_API_KEY, SERPER_API_KEY, etc.)
#   - For Character.AI:   set CAI_TOKEN=<your_token>
#   - For DeepPersona:    set DATASET_DIR=/path/to/deeppersona
#   - For OpenCharacter:  start vLLM server first (vllm serve <model> --port 8000)
#   - For ConsistentLLM:  start vLLM server first (vllm serve <model> --port 8001)
#
# Common options (apply to all baselines unless overridden per-script):
#   SAMPLE_N=10    — number of personas to sample per baseline (0 = all)
#   SEED=42        — random seed
#
# Usage:
#   bash scripts/main.sh
#   SAMPLE_N=5 bash scripts/main.sh
#   SAMPLE_N=0 DATASET_DIR=/path/to/deeppersona CAI_TOKEN=<token> bash scripts/main.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SAMPLE_N=${SAMPLE_N:-10}
export SEED=${SEED:-42}

echo "=================================================="
echo "PICON: Running all 8 persona agent baselines"
echo "SAMPLE_N=${SAMPLE_N}, SEED=${SEED}"
echo "=================================================="

# 1. Nemotron (nvidia/Nemotron-Personas-* — 7 regional HuggingFace datasets)
echo ""
echo "[1/8] Nemotron"
bash "${SCRIPT_DIR}/nemotron.sh"

# 2. Twin-2K-500 (LLM-Digital-Twin/Twin-2K-500 HuggingFace dataset)
echo ""
echo "[2/8] Twin-2K-500"
bash "${SCRIPT_DIR}/twin_2k_500.sh"

# 3. LLM-Generated (Tianyi-Lab/Personas HuggingFace dataset)
echo ""
echo "[3/8] LLM-Generated"
bash "${SCRIPT_DIR}/llm_generated.sh"

# 4. DeepPersona (requires DATASET_DIR=/path/to/deeppersona)
echo ""
echo "[4/8] DeepPersona"
if [ -z "${DATASET_DIR}" ]; then
    echo "SKIPPED: DATASET_DIR is not set. Set DATASET_DIR=/path/to/deeppersona to run this baseline."
else
    bash "${SCRIPT_DIR}/deeppersona.sh"
fi

# 5. Human Simulacra (11 fixed RAG-based characters)
echo ""
echo "[5/8] Human Simulacra"
bash "${SCRIPT_DIR}/human_simulacra.sh"

# 6. ConsistentLLM (requires vLLM server: vllm serve <model> --port 8001)
echo ""
echo "[6/8] ConsistentLLM"
bash "${SCRIPT_DIR}/consistent_llm.sh"

# 7. OpenCharacter (requires vLLM server: vllm serve <model> --port 8000)
echo ""
echo "[7/8] OpenCharacter"
bash "${SCRIPT_DIR}/opencharacter.sh"

# 8. Character.AI (requires CAI_TOKEN)
echo ""
echo "[8/8] Character.AI"
if [ -z "${CAI_TOKEN}" ] && [ -z "${CAI_API_KEY}" ]; then
    echo "SKIPPED: CAI_TOKEN is not set. Set CAI_TOKEN=<your_token> to run this baseline."
else
    bash "${SCRIPT_DIR}/characterai.sh"
fi

echo ""
echo "=================================================="
echo "All baselines completed."
echo "=================================================="
