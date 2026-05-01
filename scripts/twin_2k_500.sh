#!/bin/bash
# Baseline: Twin 2K 500
# Dataset: HuggingFace LLM-Digital-Twin/Twin-2K-500 (full_persona split)
#
# Usage:
#   bash scripts/twin_2k_500.sh                   # 기본: 10개 랜덤 샘플
#   SAMPLE_N=0 bash scripts/twin_2k_500.sh        # 전체 실행
#   SAMPLE_N=5 SEED=123 bash scripts/twin_2k_500.sh

SAMPLE_N=${SAMPLE_N:-10}
SEED=${SEED:-42}
SIMULATOR_MODEL=${SIMULATOR_MODEL:-"gemini/gemini-2.5-flash"}
BASE_PORT=${BASE_PORT:-8100}
MAX_PARALLEL=${MAX_PARALLEL:-5}

wait_for_slot() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
        wait -n 2>/dev/null || sleep 0.5
    done
}

PERSONA_LIST=$(python3 - <<EOF
from datasets import load_dataset
import random, json

random.seed(${SEED})
dataset = load_dataset("LLM-Digital-Twin/Twin-2K-500", "full_persona", split="data")

sample_n = ${SAMPLE_N}
if sample_n > 0:
    dataset = dataset.shuffle(seed=${SEED}).select(range(min(sample_n, len(dataset))))

for data in dataset:
    name = f"Twin-{data['pid']}"
    print(name + "\t" + json.dumps(data['persona_json']))
EOF
)

if [ -z "${PERSONA_LIST}" ]; then
    echo "No personas loaded."
    exit 1
fi

TOTAL=$(echo "${PERSONA_LIST}" | wc -l)
echo "Running Twin-2K-500 interviews: ${TOTAL} personas (SAMPLE_N=${SAMPLE_N}, SEED=${SEED}, MAX_PARALLEL=${MAX_PARALLEL})"

COUNT=0
while IFS=$'\t' read -r PERSONA_NAME PERSONA_JSON; do
    COUNT=$((COUNT + 1))
    PORT=$((BASE_PORT + COUNT))
    echo "[${COUNT}/${TOTAL}] ${PERSONA_NAME} (port=${PORT})"

    TMPFILE=$(mktemp /tmp/twin_XXXXXX.txt)
    python3 - <<EOF > "${TMPFILE}"
import json
from picon.env.interviewee_simulator.persona_prompt_builders import build_twin_2k_500_prompt
persona_json = json.loads(r"""${PERSONA_JSON}""")
print(build_twin_2k_500_prompt(json.dumps(persona_json, ensure_ascii=False)))
EOF

    wait_for_slot
    (
        python servers/twin_2k_500_server.py \
            --port "${PORT}" \
            --model "${SIMULATOR_MODEL}" \
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
