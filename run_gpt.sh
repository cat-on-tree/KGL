#!/bin/zsh
set -e  # 遇到错误立即退出

python src/gpt_test/biored_gpt_api.py --answer data/evaluation/model_answer/Qwen/qwen3-4b/qwen3-4b-bioRED-answer.json --result data/evaluation/gpt_test/qwen3-4b-bioRED-gpt.json --log logs/qwen3-4b-bioRED-gpt-generation.log

python src/gpt_test/biored_gpt_api.py --answer data/evaluation/model_answer/Qwen/qwen3-8b/qwen3-8b-bioRED-answer.json --result data/evaluation/gpt_test/qwen3-8b-bioRED-gpt.json --log logs/qwen3-8b-bioRED-gpt-generation.log

python src/gpt_test/biored_gpt_api.py --answer data/evaluation/model_answer/Qwen/qwen3-14b/qwen3-14b-bioRED-answer.json --result data/evaluation/gpt_test/qwen3-14b-bioRED-gpt.json --log logs/qwen3-14b-bioRED-gpt-generation.log

python src/gpt_test/biored_gpt_api.py --answer data/evaluation/model_answer/Qwen/qwen3-30b-a3b/qwen3-30b-a3b-bioRED-answer.json --result data/evaluation/gpt_test/qwen3-30b-a3b-bioRED-gpt.json --log logs/qwen3-30b-a3b-bioRED-gpt-generation.log

python src/gpt_test/biored_gpt_api.py --answer data/evaluation/model_answer/Qwen/qwen3-32b/qwen3-32b-bioRED-answer.json --result data/evaluation/gpt_test/qwen3-32b-bioRED-gpt.json --log logs/qwen3-32b-bioRED-gpt-generation.log

python src/gpt_test/biored_gpt_api.py --answer data/evaluation/model_answer/Qwen/qwen3-235b-a22b/qwen3-235b-a22b-bioRED-answer.json --result data/evaluation/gpt_test/qwen3-235b-a22b-bioRED-gpt.json --log logs/qwen3-235b-a22b-bioRED-gpt-generation.log

python src/gpt_test/biored_gpt_api.py --answer data/evaluation/model_answer/DeepSeek/deepseek-r1-distill-qwen-1.5b/deepseek-r1-distill-qwen-1.5b-bioRED-answer.json --result data/evaluation/gpt_test/deepseek-r1-distill-qwen-1.5b-bioRED-gpt.json --log logs/deepseek-r1-distill-qwen-1.5b-bioRED-gpt-generation.log

python src/gpt_test/biored_gpt_api.py --answer data/evaluation/model_answer/DeepSeek/deepseek-r1-distill-llama-8b/deepseek-r1-distill-llama-8b-bioRED-answer.json --result data/evaluation/gpt_test/deepseek-r1-distill-llama-8b-bioRED-gpt.json --log logs/deepseek-r1-distill-llama-8b-bioRED-gpt-generation.log

python src/gpt_test/biored_gpt_api.py --answer data/evaluation/model_answer/DeepSeek/deepseek-r1-distill-qwen-14b/deepseek-r1-distill-qwen-14b-bioRED-answer.json --result data/evaluation/gpt_test/deepseek-r1-distill-qwen-14b-bioRED-gpt.json --log logs/deepseek-r1-distill-qwen-14b-bioRED-gpt-generation.log

python src/gpt_test/biored_gpt_api.py --answer data/evaluation/model_answer/DeepSeek/deepseek-r1-distill-qwen-32b/deepseek-r1-distill-qwen-32b-bioRED-answer.json --result data/evaluation/gpt_test/deepseek-r1-distill-qwen-32b-bioRED-gpt.json --log logs/deepseek-r1-distill-qwen-32b-bioRED-gpt-generation.log

python src/gpt_test/biored_gpt_api.py --answer data/evaluation/model_answer/DeepSeek/deepseek-r1-distill-llama-70b/deepseek-r1-distill-llama-70b-bioRED-answer.json --result data/evaluation/gpt_test/deepseek-r1-distill-llama-70b-bioRED-gpt.json --log logs/deepseek-r1-distill-llama-70b-bioRED-gpt-generation.log

python src/gpt_test/biored_gpt_api.py --answer data/evaluation/model_answer/DeepSeek/deepseek-r1/deepseek-r1-bioRED-answer.json --result data/evaluation/gpt_test/deepseek-r1-bioRED-gpt.json --log logs/deepseek-r1-bioRED-gpt-generation.log

