#!/bin/bash
# Baseline: OpenCharacter
# Dataset: HuggingFace xywang1/OpenCharacter (Synthetic-Character split)
# Requires: vLLM server running with opencharacter-sft model
#
# Start the model server first:
#   vllm serve <your-opencharacter-model> --port 8000
#
# Usage:
#   bash scripts/opencharacter.sh                              # 기본: 10개 랜덤 샘플
#   SAMPLE_N=0 bash scripts/opencharacter.sh                   # 전체 실행
#   VLLM_BASE=http://localhost:8000/v1 SAMPLE_N=5 bash scripts/opencharacter.sh

VLLM_BASE=${VLLM_BASE:-"http://localhost:8000/v1"}
VLLM_MODEL=${VLLM_MODEL:-"anonymous/opencharacter-sft-llama-3-8b-instruct"}
BASE_PORT=${BASE_PORT:-8100}
SAMPLE_N=${SAMPLE_N:-10}
SEED=${SEED:-42}
MAX_PARALLEL=${MAX_PARALLEL:-5}

wait_for_slot() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
        wait -n 2>/dev/null || sleep 0.5
    done
}

PERSONA_LIST=$(python3 - <<EOF
from datasets import load_dataset
import re, json

dataset = load_dataset("xywang1/OpenCharacter", "Synthetic-Character", split="train")

sample_n = ${SAMPLE_N}
if sample_n > 0:
    dataset = dataset.shuffle(seed=${SEED}).select(range(min(sample_n, len(dataset))))

for data in dataset:
    name_match = re.match(r"Name:\s(.*)\n", data['character'])
    if not name_match:
        continue
    name = name_match.group(1).strip()
    persona = {"persona": data["persona"], "profile": data["character"]}
    print(name + "\t" + json.dumps(persona, ensure_ascii=False))
EOF
)

if [ -z "${PERSONA_LIST}" ]; then
    echo "No personas loaded."
    exit 1
fi

TOTAL=$(echo "${PERSONA_LIST}" | wc -l)
echo "Running OpenCharacter interviews: ${TOTAL} personas (SAMPLE_N=${SAMPLE_N}, SEED=${SEED}, MAX_PARALLEL=${MAX_PARALLEL})"
echo "vLLM: ${VLLM_BASE} / ${VLLM_MODEL}"

COUNT=0
while IFS=$'\t' read -r PERSONA_NAME PERSONA_JSON; do
    COUNT=$((COUNT + 1))
    PORT=$((BASE_PORT + COUNT))
    echo "[${COUNT}/${TOTAL}] ${PERSONA_NAME} (port=${PORT})"

    TMPFILE=$(mktemp /tmp/openchar_XXXXXX.txt)
    python3 - <<EOF
import json
from picon.env.interviewee_simulator.persona_prompt_builders import build_opencharacter_prompt
data = json.loads(r"""${PERSONA_JSON}""")
print(build_opencharacter_prompt(data["persona"], data["profile"]))
EOF
    > "${TMPFILE}"

    wait_for_slot
    (
        python servers/opencharacter_server.py \
            --port "${PORT}" \
            --vllm_base "${VLLM_BASE}" \
            --vllm_model "${VLLM_MODEL}" \
            --persona_file "${TMPFILE}" \
            --name "${PERSONA_NAME}" &
        SERVER_PID=$!

        for i in $(seq 1 30); do
            if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
                break
            fi
            sleep 1
        done

        picon \
            --agent_api_base "http://localhost:${PORT}/v1" \
            --agent_name "${PERSONA_NAME}" \
            --questioner_model gpt-5 \
            --extractor_model gpt-5.1 \
            --web_search_model gpt-5 \
            --evaluator_model gemini/gemini-2.5-flash \
            --nhd_model gpt-5-nano \
            --num_turns 50 \
            --num_sessions 2 \
            --do_eval \
            --log_to_file

        kill "${SERVER_PID}" 2>/dev/null
        wait "${SERVER_PID}" 2>/dev/null
        rm -f "${TMPFILE}"
    ) &

done <<< "${PERSONA_LIST}"
wait

echo "Done. ${TOTAL} personas completed."
