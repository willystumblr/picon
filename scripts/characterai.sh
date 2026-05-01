#!/bin/bash
# Baseline: Character.AI
# Characters: picon/env/personas/characterai.json (10명 고정)
# CAI_TOKEN 환경변수 필요 (Character.AI session token)
#
# Usage:
#   CAI_TOKEN=<token> bash scripts/characterai.sh
#   CAI_TOKEN=<token> SAMPLE_N=5 bash scripts/characterai.sh
#   CAI_TOKEN=<token> SAMPLE_N=0 bash scripts/characterai.sh   # 전체 실행
#   CAI_TOKEN=<token> SAMPLE_N=3 SEED=7 bash scripts/characterai.sh

SERVER_HOST=${SERVER_HOST:-"localhost"}
BASE_PORT=${BASE_PORT:-8100}   # 캐릭터마다 BASE_PORT+N 포트 사용
SAMPLE_N=${SAMPLE_N:-1}
SEED=${SEED:-42}
PERSONAS_FILE=${PERSONAS_FILE:-"picon/env/personas/characterai.json"}
MAX_PARALLEL=${MAX_PARALLEL:-5}

if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi
CAI_TOKEN=${CAI_TOKEN:-${CAI_API_KEY:-""}}

if [ -z "${CAI_TOKEN}" ]; then
    echo "ERROR: CAI_TOKEN (or CAI_API_KEY) is required. Set it in .env or via: CAI_TOKEN=<your_token> bash scripts/characterai.sh"
    exit 1
fi

wait_for_slot() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
        wait -n 2>/dev/null || sleep 0.5
    done
}

PERSONA_LIST=$(python3 - <<EOF
import json, random
random.seed(${SEED})

personas = json.load(open("${PERSONAS_FILE}"))
sample_n = ${SAMPLE_N}
if sample_n > 0:
    personas = random.sample(personas, min(sample_n, len(personas)))

for p in personas:
    print(p["character_name"] + "\t" + p["character_id"])
EOF
)

if [ -z "${PERSONA_LIST}" ]; then
    echo "No personas loaded from ${PERSONAS_FILE}"
    exit 1
fi

TOTAL=$(echo "${PERSONA_LIST}" | wc -l)
echo "Running Character.AI interviews: ${TOTAL} personas (SAMPLE_N=${SAMPLE_N}, SEED=${SEED}, MAX_PARALLEL=${MAX_PARALLEL})"

COUNT=0
while IFS=$'\t' read -r CHARACTER_NAME CHARACTER_ID; do
    COUNT=$((COUNT + 1))
    PORT=$((BASE_PORT + COUNT))
    echo "[${COUNT}/${TOTAL}] ${CHARACTER_NAME} (character_id=${CHARACTER_ID}, port=${PORT})"

    wait_for_slot
    (
        # 이 캐릭터 전용 서버 시작
        python servers/characterai_server.py \
            --port "${PORT}" \
            --character_id "${CHARACTER_ID}" \
            --user_id "${CAI_TOKEN}" &
        SERVER_PID=$!

        # 서버 준비 대기
        for i in $(seq 1 30); do
            if curl -sf "http://${SERVER_HOST}:${PORT}/" > /dev/null 2>&1; then
                break
            fi
            sleep 1
        done

        picon \
            --agent_api_base "http://${SERVER_HOST}:${PORT}/v1" \
            --agent_name "${CHARACTER_NAME}" \
            --questioner_model gpt-5 \
            --extractor_model gpt-5.1 \
            --web_search_model gpt-5 \
            --evaluator_model gemini-2.5-flash \
            --nhd_model gpt-5-nano \
            --num_turns 50 \
            --num_sessions 2 \
            --do_eval \
            --log_to_file

        kill "${SERVER_PID}" 2>/dev/null
        wait "${SERVER_PID}" 2>/dev/null
    ) &

done <<< "${PERSONA_LIST}"
wait

echo "Done. ${TOTAL} personas completed."
