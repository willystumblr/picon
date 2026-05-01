#!/bin/bash
# Baseline: Human Simulacra
# Characters: 11명 고정 (RAG 기반 에이전트)
# Requires: human_simulacra_server.py (자동 기동됨)
#
# Usage:
#   bash scripts/human_simulacra.sh                   # 기본: 전체 11명 중 10명 랜덤 샘플
#   SAMPLE_N=0 bash scripts/human_simulacra.sh        # 전체 11명 실행
#   SAMPLE_N=3 SEED=7 bash scripts/human_simulacra.sh
#   SIMULATOR_MODEL=gemini/gemini-2.5-flash bash scripts/human_simulacra.sh

BASE_PORT=${BASE_PORT:-8100}
SAMPLE_N=${SAMPLE_N:-10}
SEED=${SEED:-42}
SIMULATOR_MODEL=${SIMULATOR_MODEL:-"gemini/gemini-2.5-flash"}
MAX_PARALLEL=${MAX_PARALLEL:-3}   # RAG 로딩이 무거우므로 낮게 설정

wait_for_slot() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
        wait -n 2>/dev/null || sleep 0.5
    done
}

ALL_CHARACTERS=(
    "Mary Jones"
    "Haley Collins"
    "Sara Ochoa"
    "James Jones"
    "Tami Clark"
    "Michael Miller"
    "Kevin Kelly"
    "Erica Walker"
    "Leslie Nichols"
    "Robert Scott"
    "Marsh Zhaleh"
)

CHARACTER_LIST=$(python3 - <<EOF
import random
random.seed(${SEED})
chars = [$(printf '"%s",' "${ALL_CHARACTERS[@]}")]
sample_n = ${SAMPLE_N}
if sample_n > 0:
    chars = random.sample(chars, min(sample_n, len(chars)))
for c in chars:
    print(c)
EOF
)

if [ -z "${CHARACTER_LIST}" ]; then
    echo "No characters selected."
    exit 1
fi

TOTAL=$(echo "${CHARACTER_LIST}" | wc -l)
echo "Running Human Simulacra interviews: ${TOTAL} characters (SAMPLE_N=${SAMPLE_N}, SEED=${SEED}, MAX_PARALLEL=${MAX_PARALLEL})"

COUNT=0
while IFS= read -r CHARACTER_NAME; do
    COUNT=$((COUNT + 1))
    PORT=$((BASE_PORT + COUNT))
    echo "[${COUNT}/${TOTAL}] ${CHARACTER_NAME} (port=${PORT})"

    wait_for_slot
    (
        python servers/human_simulacra_server.py \
            --port "${PORT}" \
            --character_name "${CHARACTER_NAME}" \
            --model "${SIMULATOR_MODEL}" &
        SERVER_PID=$!

        # RAG 로딩 시간 고려하여 최대 60초 대기
        for i in $(seq 1 60); do
            if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
                break
            fi
            sleep 1
        done

        picon \
            --agent_api_base "http://localhost:${PORT}/v1" \
            --agent_name "${CHARACTER_NAME}" \
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
    ) &

done <<< "${CHARACTER_LIST}"
wait

echo "Done. ${TOTAL} characters completed."
