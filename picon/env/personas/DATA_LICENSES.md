# Data Licenses

This directory contains persona data used for the PICON paper benchmark.
Each dataset is subject to its own license terms as described below.

---

## characterai.json

- **Contents**: List of character IDs and names from [Character.AI](https://character.ai/)
- **License**: No open license — governed by [Character.AI Terms of Service](https://policies.character.ai/tos)
- **Notes**: Contains only character identifiers (ID + name), not conversation data. Redistribution of character content or conversation logs is prohibited under the ToS.

---

## deeppersona/

- **Source**: DeepPersona — [arXiv:2511.07338](https://arxiv.org/abs/2511.07338)
- **GitHub**: https://github.com/thzva/Deeppersona
- **HuggingFace**: https://huggingface.co/datasets/THzva/deeppersona_dataset
- **Data License**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Code License**: MIT
- **Commercial use**: Allowed
- **Redistribution**: Allowed with attribution

---

## human_simulacra/

- **Source**: Human Simulacra — [arXiv:2402.18180](https://arxiv.org/abs/2402.18180) (ICLR 2025)
- **GitHub**: https://github.com/hasakiXie123/Human-Simulacra
- **Data License**: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- **Code License**: Apache-2.0
- **Commercial use**: Not allowed (NC)
- **Redistribution**: Allowed for non-commercial use under the same license (SA)

---

## consistent_llm_personas.jsonl

- **Source**: Derived from ConsistentLLM — [NeurIPS 2025](https://github.com/abdulhaim/consistent-LLMs)
- **GitHub**: https://github.com/abdulhaim/consistent-LLMs
- **License**: No license file in the repository
- **Notes**: This file is **not included** in the repository. Generate it locally using:
  ```bash
  git clone https://github.com/abdulhaim/consistent-LLMs
  python scripts/build_consistent_llm_personas.py \
      --personas_file consistent-LLMs/chatting/config_chatting_personas.json \
      --config_file   consistent-LLMs/chatting/config_chatting.json \
      --output_file   picon/env/personas/consistent_llm_personas.jsonl
  ```
  The source data (`config_chatting_personas.json`) has no explicit license. Contact the authors before any redistribution.
