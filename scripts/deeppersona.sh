#!/bin/bash
# Baseline: DeepPersona
# Dataset: profile_*.json files, each containing ~60 profiles (Profile_R*_A* keys)
#
# Set the dataset directory via DATASET_DIR environment variable:
#   export DATASET_DIR="/path/to/deeppersona"
#
# Usage:
#   DATASET_DIR=/path/to/deeppersona bash scripts/deeppersona.sh   # default: 10 random samples
#   SAMPLE_N=0 bash scripts/deeppersona.sh                         # run all (no sampling)
#   SAMPLE_N=5 bash scripts/deeppersona.sh                         # sample 5 personas
#   SEED=123 bash scripts/deeppersona.sh                           # set random seed

if [ -f ".env" ]; then
  set -a; source .env; set +a
fi

if [ -z "${DATASET_DIR}" ]; then
    echo "Error: DATASET_DIR is not set."
    echo "Usage: DATASET_DIR=/path/to/deeppersona bash scripts/deeppersona.sh"
    exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
    echo "Error: DATASET_DIR does not exist: ${DATASET_DIR}"
    exit 1
fi

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

# collect all (file, profile_key) pairs then sample
PERSONA_LIST=$(python3 - <<EOF
import json, glob, os, random

random.seed(${SEED})
dataset_dir = "${DATASET_DIR}"
files = sorted(glob.glob(os.path.join(dataset_dir, "*.json")))

pairs = []
for f in files:
    data = json.load(open(f))
    for key in data.keys():
        pairs.append((f, key))

sample_n = ${SAMPLE_N}
if sample_n > 0:
    pairs = random.sample(pairs, min(sample_n, len(pairs)))

for f, key in pairs:
    print(f"{f}\t{key}")
EOF
)

if [ -z "${PERSONA_LIST}" ]; then
    echo "No personas found in ${DATASET_DIR}"
    exit 1
fi

TOTAL=$(echo "${PERSONA_LIST}" | wc -l)
echo "Running DeepPersona interviews: ${TOTAL} personas (SAMPLE_N=${SAMPLE_N}, SEED=${SEED}, MAX_PARALLEL=${MAX_PARALLEL})"

COUNT=0
while IFS=$'\t' read -r FILEPATH PROFILE_KEY; do
    COUNT=$((COUNT + 1))
    PORT=$((BASE_PORT + COUNT))
    echo "[${COUNT}/${TOTAL}] ${PROFILE_KEY} (${FILEPATH}, port=${PORT})"

    TMPFILE=$(mktemp /tmp/deeppersona_XXXXXX.txt)
    python3 - <<EOF > "${TMPFILE}"
import json
from picon.env.interviewee_simulator.persona_prompt_builders import build_deeppersona_prompt
data = json.load(open("${FILEPATH}"))
profile = data.get("${PROFILE_KEY}", data)
print(build_deeppersona_prompt(profile))
EOF

    wait_for_slot
    (
        python servers/deeppersona_server.py \
            --port "${PORT}" \
            --model "${SIMULATOR_MODEL}" \
            --persona_file "${TMPFILE}" \
            --name "${PROFILE_KEY}" &
        SERVER_PID=$!

        for i in $(seq 1 30); do
            if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
                break
            fi
            sleep 1
        done

        picon \
            --agent_api_base "http://localhost:${PORT}/v1" \
            --agent_name "${PROFILE_KEY}" \
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
