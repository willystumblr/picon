#!/bin/bash
# Baseline: Nemotron Personas (all regions combined)
# Datasets:
#   nvidia/Nemotron-Personas-USA
#   nvidia/Nemotron-Personas-Korea
#   nvidia/Nemotron-Personas-Singapore
#   nvidia/Nemotron-Personas-France
#   nvidia/Nemotron-Personas-India
#   nvidia/Nemotron-Personas-Japan
#   nvidia/Nemotron-Personas-Brazil
#
# Sampling: at least 1 per dataset, remainder filled randomly across all datasets.
#
# Usage:
#   bash scripts/nemotron.sh                        # default: 10 personas (1 per region + 3 extra)
#   SAMPLE_N=7 bash scripts/nemotron.sh             # exactly 1 per region
#   SAMPLE_N=14 SEED=123 bash scripts/nemotron.sh

SAMPLE_N=${SAMPLE_N:-10}
SEED=${SEED:-42}
SIMULATOR_MODEL=${SIMULATOR_MODEL:-"gemini/gemini-3-flash-preview"}
BASE_PORT=${BASE_PORT:-8100}
MAX_PARALLEL=${MAX_PARALLEL:-5}

if [ -f ".env" ]; then
  set -a; source .env; set +a
fi

wait_for_slot() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
        wait -n 2>/dev/null || sleep 0.5
    done
}

PERSONAS_JSON=${PERSONAS_JSON:-"picon/env/personas/nemotron_personas.json"}

python3 - <<EOF
from datasets import load_dataset
import random, json
from picon.env.interviewee_simulator.persona_prompt_builders import build_nemotron_prompt

DATASETS = [
    ("nvidia/Nemotron-Personas-USA",       "usa"),
    ("nvidia/Nemotron-Personas-Korea",     "kor"),
    ("nvidia/Nemotron-Personas-Singapore", "sgp"),
    ("nvidia/Nemotron-Personas-France",    "fra"),
    ("nvidia/Nemotron-Personas-India",     "ind"),
    ("nvidia/Nemotron-Personas-Japan",     "jpn"),
    ("nvidia/Nemotron-Personas-Brazil",    "bra"),
]

rng = random.Random(${SEED})
sample_n = ${SAMPLE_N}
n_groups = len(DATASETS)

quotas = [1] * n_groups
remainder = max(0, sample_n - n_groups)
for _ in range(remainder):
    quotas[rng.randrange(n_groups)] += 1

selected = []
for (repo, region), quota in zip(DATASETS, quotas):
    split_type = "en_IN" if region == "ind" else "train"
    ds = load_dataset(repo, split=split_type)
    ds = ds.shuffle(seed=${SEED}).select(range(min(quota, len(ds))))
    for d in ds:
        selected.append((region, dict(d)))

rng.shuffle(selected)

records = []
for region, d in selected:
    uid = d.get("uuid", "unknown")
    records.append({
        "name":   f"Nemotron-{region.upper()}-{uid}",
        "region": region,
        "uuid":   uid,
        "prompt": build_nemotron_prompt(d),
    })

json.dump(records, open("${PERSONAS_JSON}", "w"), ensure_ascii=False, indent=2)
print(f"Saved {len(records)} personas to ${PERSONAS_JSON}")
EOF

TOTAL=$(python3 -c "import json; print(len(json.load(open('${PERSONAS_JSON}'))))")

if [ "${TOTAL}" -eq 0 ]; then
    echo "No personas loaded."
    exit 1
fi

echo "Running Nemotron (all regions) interviews: ${TOTAL} personas (SAMPLE_N=${SAMPLE_N}, SEED=${SEED}, MAX_PARALLEL=${MAX_PARALLEL})"

COUNT=0
while IFS= read -r PERSONA_NAME; do
    IFS= read -r TMPFILE
    COUNT=$((COUNT + 1))
    PORT=$((BASE_PORT + COUNT))
    echo "[${COUNT}/${TOTAL}] ${PERSONA_NAME} (port=${PORT})"

    wait_for_slot
    (
        python servers/nemotron_server.py \
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

done < <(python3 - <<EOF
import json, tempfile, os
records = json.load(open("${PERSONAS_JSON}"))
for r in records:
    fd, path = tempfile.mkstemp(prefix="nemotron_", suffix=".txt", dir="/tmp")
    with os.fdopen(fd, "w") as f:
        f.write(r["prompt"])
    print(r["name"])
    print(path)
EOF
)
wait

echo "Done. ${TOTAL} personas completed."
