#!/bin/zsh

set -e  # 遇到错误立即退出

python src/gpt_result/gpt_result.py --test data/evaluation/gpt_test/DeepSeek/deepseek-r1/deepseek-r1-bioASQ-gpt.json --result data/evaluation/gpt_result/deepseek-r1-bioASQ-gpt.txt

python src/gpt_result/gpt_result.py --test data/evaluation/gpt_test/DeepSeek/deepseek-r1-distill-llama-8b/deepseek-r1-distill-llama-8b-bioASQ-gpt.json --result data/evaluation/gpt_result/deepseek-r1-distill-llama-8b-bioASQ-gpt.txt

python src/gpt_result/gpt_result.py --test data/evaluation/gpt_test/DeepSeek/deepseek-r1-distill-llama-70b/deepseek-r1-distill-llama-70b-bioASQ-gpt.json --result data/evaluation/gpt_result/deepseek-r1-distill-llama-70b-bioASQ-gpt.txt

python src/gpt_result/gpt_result.py --test data/evaluation/gpt_test/DeepSeek/deepseek-r1-distill-qwen-1.5b/deepseek-r1-distill-qwen-1.5b-bioASQ-gpt.json --result data/evaluation/gpt_result/deepseek-r1-distill-qwen-1.5b-bioASQ-gpt.txt

python src/gpt_result/gpt_result.py --test data/evaluation/gpt_test/DeepSeek/deepseek-r1-distill-qwen-14b/deepseek-r1-distill-qwen-14b-bioASQ-gpt.json --result data/evaluation/gpt_result/deepseek-r1-distill-qwen-14b-bioASQ-gpt.txt

python src/gpt_result/gpt_result.py --test data/evaluation/gpt_test/DeepSeek/deepseek-r1-distill-qwen-32b/deepseek-r1-distill-qwen-32b-bioASQ-gpt.json --result data/evaluation/gpt_result/deepseek-r1-distill-qwen-32b-bioASQ-gpt.txt

python src/gpt_result/gpt_result.py --test data/evaluation/gpt_test/Qwen/qwen3-4b/qwen3-4b-bioASQ-gpt.json --result data/evaluation/gpt_result/qwen3-4b-bioASQ-gpt.txt

python src/gpt_result/gpt_result.py --test data/evaluation/gpt_test/Qwen/qwen3-8b/qwen3-8b-bioASQ-gpt.json --result data/evaluation/gpt_result/qwen3-8b-bioASQ-gpt.txt

python src/gpt_result/gpt_result.py --test data/evaluation/gpt_test/Qwen/qwen3-14b/qwen3-14b-bioASQ-gpt.json --result data/evaluation/gpt_result/qwen3-14b-bioASQ-gpt.txt

python src/gpt_result/gpt_result.py --test data/evaluation/gpt_test/Qwen/qwen3-30b-a3b/qwen3-30b-a3b-bioASQ-gpt.json --result data/evaluation/gpt_result/qwen3-30b-a3b-bioASQ-gpt.txt

python src/gpt_result/gpt_result.py --test data/evaluation/gpt_test/Qwen/qwen3-32b/qwen3-32b-bioASQ-gpt.json --result data/evaluation/gpt_result/qwen3-32b-bioASQ-gpt.txt

python src/gpt_result/gpt_result.py --test data/evaluation/gpt_test/Qwen/qwen3-235b-a22b/qwen3-235b-a22b-bioASQ-gpt.json --result data/evaluation/gpt_result/qwen3-235b-a22b-bioASQ-gpt.txt
