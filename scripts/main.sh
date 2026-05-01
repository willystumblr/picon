#!/bin/bash
python main.py \
    --baseline_name opencharacter \
    --questioner_model gpt-5 \
    --extractor_model gpt-5.1 \
    --web_search_model gpt-5 \
    --num_turns 50 \
    --nhd_model gpt-5-nano \
    --evaluator_model gpt-5.1 \
    --simulator_model hosted_vllm/anonymous/opencharacter-sft-llama-3-8b-instruct --simulator_host localhost--simulator_port 8123 \
    --do_sample

python main.py \
    --baseline_name llm_generated \
    --questioner_model hosted_vllm/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8 \
    --questioner_port 8899 \
    --extractor_model hosted_vllm/Qwen/Qwen3-Next-80B-A3B-Thinking \
    --extractor_port 8001 \
    --web_search_model hosted_vllm/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8  \
    --web_search_port 8899 \
    --num_turns 50 \
    --nhd_model gpt-5-nano \
    --evaluator_model hosted_vllm/Qwen/Qwen3-Next-80B-A3B-Thinking \
    --evaluator_port 8899 \
    --log_to_file \
    --simulator_model gemini/gemini-3-flash-preview  \
    --do_sample

python main.py \
    --baseline_name characterai \
    --questioner_model hosted_vllm/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8 \
    --questioner_port 8900 \
    --extractor_model hosted_vllm/Qwen/Qwen3-Next-80B-A3B-Thinking \
    --extractor_port 8899 \
    --web_search_model hosted_vllm/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8  \
    --web_search_port 8900\
    --num_turns 50 \
    --nhd_model gpt-5-nano \
    --evaluator_model hosted_vllm/Qwen/Qwen3-Next-80B-A3B-Thinking \
    --evaluator_port 8899 \
    --log_to_file



python main.py \
    --baseline_name twin_2k_500 \
    --questioner_model hosted_vllm/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8 \
    --questioner_port 8900 \
    --extractor_model hosted_vllm/Qwen/Qwen3-Next-80B-A3B-Thinking \
    --extractor_port 8899 \
    --web_search_model hosted_vllm/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8  \
    --web_search_port 8900 \
    --num_turns 50 \
    --nhd_model gpt-5-nano \
    --simulator_model gemini/gemini-3-flash-preview \
    --evaluator_model hosted_vllm/Qwen/Qwen3-Next-80B-A3B-Thinking \
    --evaluator_port 8899 \
    --log_to_file \
    --do_sample

python main.py \
    --baseline_name deeppersona \
    --questioner_model gpt-5  \
    --extractor_model gpt-5.1 \
    --web_search_model gpt-5  \
    --num_turns 50 \
    --nhd_model gpt-5-nano \
    --simulator_model gemini/gemini-3-flash-preview \
    --evaluator_model gpt-5.1 \
    --log_to_file \
    --do_sample

python main.py \
    --baseline_name deeppersona \
    --questioner_model hosted_vllm/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8 \
    --questioner_port 8900 \
    --extractor_model hosted_vllm/Qwen/Qwen3-Next-80B-A3B-Thinking \
    --extractor_port 8899 \
    --web_search_model hosted_vllm/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8  \
    --web_search_port 8900 \
    --num_turns 50 \
    --nhd_model gpt-5-nano \
    --evaluator_model hosted_vllm/Qwen/Qwen3-Next-80B-A3B-Thinking \
    --evaluator_port 8899 \
    --simulator_model gemini/gemini-3-flash-preview \
    --log_to_file

python main.py \
    --baseline_name human_simulacra \
    --questioner_model hosted_vllm/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8 \
    --questioner_port 8900 \
    --extractor_model hosted_vllm/Qwen/Qwen3-Next-80B-A3B-Thinking \
    --extractor_port 8899 \
    --web_search_model hosted_vllm/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8  \
    --web_search_port 8900 \
    --num_turns 50 \
    --nhd_model gpt-5-nano \
    --simulator_model gemini/gemini-3-flash-preview \
    --evaluator_model hosted_vllm/Qwen/Qwen3-Next-80B-A3B-Thinking \
    --evaluator_port 8899 \
    --num_sessions 2 \
    --log_to_file


python main.py \
    --baseline_name consistent_llm \
    --questioner_model hosted_vllm/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8 \
    --questioner_port 8900 \
    --extractor_model hosted_vllm/Qwen/Qwen3-Next-80B-A3B-Thinking \
    --extractor_port 8899 \
    --web_search_model hosted_vllm/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8  \
    --web_search_port 8900 \
    --num_turns 50 \
    --nhd_model gpt-5-nano \
    --evaluator_model hosted_vllm/Qwen/Qwen3-Next-80B-A3B-Thinking \
    --evaluator_port 8899 \
    --simulator_model hosted_vllm/anonymous/consistent_llm_llama-8b-sft-ppo-prompt --simulator_port 8001 \
    --simulator_host localhost \
    --do_sample 


python main.py \
    --baseline_name opencharacter \
    --questioner_model hosted_vllm/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8 \
    --questioner_port 8900 \
    --extractor_model hosted_vllm/Qwen/Qwen3-Next-80B-A3B-Thinking \
    --extractor_port 8899 \
    --web_search_model hosted_vllm/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8  \
    --web_search_port 8900 \
    --num_turns 50 \
    --nhd_model gpt-5-nano \
    --evaluator_model hosted_vllm/Qwen/Qwen3-Next-80B-A3B-Thinking \
    --evaluator_port 8899 \
    --simulator_model hosted_vllm/anonymous/opencharacter-sft-llama-3-8b-instruct \
    --simulator_port 8123 \
    --simulator_host localhost \
    --do_sample
