#!/bin/bash
# Baseline: ConsistentLLM
# Dataset: picon/env/personas/consistent_llm_personas.jsonl
#   - 각 줄: persona, name, counterpart_name, instruction, model_path 필드 포함 JSON
#
# Requires: ConsistentLLM fine-tuned model served via vLLM
#   vllm serve <model_path> --port 8001
#
# Usage:
#   bash scripts/consistent_llm.sh                          # 기본: 10개 랜덤 샘플
#   SAMPLE_N=0 bash scripts/consistent_llm.sh              # 전체 실행
#   SAMPLE_N=5 SEED=123 bash scripts/consistent_llm.sh
#   SIMULATOR_MODEL=hosted_vllm/my-model bash scripts/consistent_llm.sh
#   SIMULATOR_PORT=8002 bash scripts/consistent_llm.sh

SAMPLE_N=${SAMPLE_N:-10}
SEED=${SEED:-42}
MAX_PARALLEL=${MAX_PARALLEL:-5}
SIMULATOR_HOST=${SIMULATOR_HOST:-"localhost"}
SIMULATOR_PORT=${SIMULATOR_PORT:-8001}
SIMULATOR_MODEL=${SIMULATOR_MODEL:-"hosted_vllm/anonymous/consistent_llm_llama-8b-sft-ppo-prompt"}
PERSONAS_FILE=${PERSONAS_FILE:-"picon/env/personas/consistent_llm_personas.jsonl"}

if [ -f ".env" ]; then
  set -a; source .env; set +a
fi

wait_for_slot() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
        wait -n 2>/dev/null || sleep 0.5
    done
}

if [ ! -f "${PERSONAS_FILE}" ]; then
    echo "Personas file not found: ${PERSONAS_FILE}"
    exit 1
fi

PERSONA_LIST=$(python3 - <<EOF
import json, random

random.seed(${SEED})
personas = []
with open("${PERSONAS_FILE}") as f:
    for line in f:
        line = line.strip()
        if line:
            personas.append(line)

sample_n = ${SAMPLE_N}
if sample_n > 0:
    personas = random.sample(personas, min(sample_n, len(personas)))

for p in personas:
    data = json.loads(p)
    name = data.get("name", "Unknown")
    # Print name and full JSON on one line, tab-separated
    print(name + "\t" + p)
EOF
)

if [ -z "${PERSONA_LIST}" ]; then
    echo "No personas found in ${PERSONAS_FILE}"
    exit 1
fi

TOTAL=$(echo "${PERSONA_LIST}" | wc -l)
echo "Running ConsistentLLM interviews: ${TOTAL} personas (SAMPLE_N=${SAMPLE_N}, SEED=${SEED}, MAX_PARALLEL=${MAX_PARALLEL})"
echo "Simulator: ${SIMULATOR_MODEL} @ ${SIMULATOR_HOST}:${SIMULATOR_PORT}"

COUNT=0
while IFS=$'\t' read -r PERSONA_NAME PERSONA_JSON; do
    COUNT=$((COUNT + 1))
    echo "[${COUNT}/${TOTAL}] ${PERSONA_NAME}"

    wait_for_slot
    (
        picon \
            --baseline_name consistent_llm \
            --agent_name "${PERSONA_NAME}" \
            --agent_persona "${PERSONA_JSON}" \
            --simulator_model "${SIMULATOR_MODEL}" \
            --simulator_host "${SIMULATOR_HOST}" \
            --simulator_port "${SIMULATOR_PORT}" \
            --questioner_model gpt-5 \
            --extractor_model gpt-5.1 \
            --web_search_model gpt-5 \
            --evaluator_model gemini/gemini-2.5-flash \
            --nhd_model gpt-5-nano \
            --num_turns 50 \
            --num_sessions 2 \
            --do_eval \
            --log_to_file
    ) &

done <<< "${PERSONA_LIST}"
wait

echo "Done. ${TOTAL} personas completed."
