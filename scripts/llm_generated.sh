#!/bin/bash
# Baseline: LLM Generated Persona
# Dataset: HuggingFace Tianyi-Lab/Personas
# Paper: https://arxiv.org/abs/2503.16527
#
# Usage:
#   bash scripts/llm_generated.sh                              # default: 10 random samples
#   SAMPLE_N=0 bash scripts/llm_generated.sh                   # run all
#   SAMPLE_N=5 SEED=123 PERSONA_TYPE=objective bash scripts/llm_generated.sh
#
# PERSONA_TYPE: descriptive (default) | subjective | objective | meta

SAMPLE_N=${SAMPLE_N:-1}
SEED=${SEED:-42}
PERSONA_TYPE=${PERSONA_TYPE:-"descriptive"}
SIMULATOR_MODEL=${SIMULATOR_MODEL:-"gemini/gemini-3-flash-preview"}
BASE_PORT=${BASE_PORT:-8100}
MAX_PARALLEL=${MAX_PARALLEL:-5}

wait_for_slot() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
        wait -n 2>/dev/null || sleep 0.5
    done
}

PERSONA_LIST=$(python3 - <<EOF
from datasets import load_dataset
import json

dataset = load_dataset("Tianyi-Lab/Personas", split="train")

sample_n = ${SAMPLE_N}
if sample_n > 0:
    dataset = dataset.shuffle(seed=${SEED}).select(range(min(sample_n, len(dataset))))

preferred_prefixes = ["Llama-3.1-70B-Instruct"]
available_prefixes = [
    col[:-len("_descriptive_persona")]
    for col in dataset.column_names
    if col.endswith("_descriptive_persona")
]
selected_prefix = next((p for p in preferred_prefixes if p in available_prefixes), available_prefixes[0] if available_prefixes else None)
if selected_prefix is None:
    raise ValueError("No persona columns found in Tianyi-Lab/Personas.")

for data in dataset:
    persona_number = data.get("persona_number", "unknown")
    name = f"LLM-Persona-{persona_number}"
    persona = {
        "descriptive_persona": data.get(f"{selected_prefix}_descriptive_persona", ""),
        "objective_table_persona": data.get(f"{selected_prefix}_objective_table_persona", ""),
        "subjective_table_persona": data.get(f"{selected_prefix}_subjective_table_persona", ""),
        "meta_persona": data.get("meta_persona", ""),
    }
    print(name + "\t" + json.dumps(persona, ensure_ascii=False))
EOF
)

if [ -z "${PERSONA_LIST}" ]; then
    echo "No personas loaded."
    exit 1
fi

TOTAL=$(echo "${PERSONA_LIST}" | wc -l)
echo "Running LLM-Generated interviews: ${TOTAL} personas (SAMPLE_N=${SAMPLE_N}, SEED=${SEED}, PERSONA_TYPE=${PERSONA_TYPE}, MAX_PARALLEL=${MAX_PARALLEL})"

COUNT=0
while IFS=$'\t' read -r PERSONA_NAME PERSONA_JSON; do
    COUNT=$((COUNT + 1))
    PORT=$((BASE_PORT + COUNT))
    echo "[${COUNT}/${TOTAL}] ${PERSONA_NAME} (port=${PORT})"

    TMPFILE=$(mktemp /tmp/llmgen_XXXXXX.txt)
    python3 - <<EOF > "${TMPFILE}"
import json
from picon.env.interviewee_simulator.persona_prompt_builders import build_llm_generated_prompt
persona = json.loads(r"""${PERSONA_JSON}""")
print(build_llm_generated_prompt(persona, persona_type="${PERSONA_TYPE}"))
EOF

    wait_for_slot
    (
        python servers/llm_generated_server.py \
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
