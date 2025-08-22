#!/bin/zsh
set -e  # 遇到错误立即退出

python src/gpt_test/bioASQ_gpt_api.py --answer data/evaluation/model_answer/Qwen/qwen3-4b/qwen3-4b-bioASQ-answer.json --output data/evaluation/gpt_test/qwen3-4b-bioASQ-gpt.json --log logs/qwen3-4b-bioASQ-gpt-generation.log

python src/gpt_test/bioASQ_gpt_api.py --answer data/evaluation/model_answer/Qwen/qwen3-8b/qwen3-8b-bioASQ-answer.json --output data/evaluation/gpt_test/qwen3-8b-bioASQ-gpt.json --log logs/qwen3-8b-bioASQ-gpt-generation.log

python src/gpt_test/bioASQ_gpt_api.py --answer data/evaluation/model_answer/Qwen/qwen3-14b/qwen3-14b-bioASQ-answer.json --output data/evaluation/gpt_test/qwen3-14b-bioASQ-gpt.json --log logs/qwen3-14b-bioASQ-gpt-generation.log

python src/gpt_test/bioASQ_gpt_api.py --answer data/evaluation/model_answer/Qwen/qwen3-30b-a3b/qwen3-30b-a3b-bioASQ-answer.json --output data/evaluation/gpt_test/qwen3-30b-a3b-bioASQ-gpt.json --log logs/qwen3-30b-a3b-bioASQ-gpt-generation.log

python src/gpt_test/bioASQ_gpt_api.py --answer data/evaluation/model_answer/Qwen/qwen3-32b/qwen3-32b-bioASQ-answer.json --output data/evaluation/gpt_test/qwen3-32b-bioASQ-gpt.json --log logs/qwen3-32b-bioASQ-gpt-generation.log

python src/gpt_test/bioASQ_gpt_api.py --answer data/evaluation/model_answer/Qwen/qwen3-235b-a22b/qwen3-235b-a22b-bioASQ-answer.json --output data/evaluation/gpt_test/qwen3-235b-a22b-bioASQ-gpt.json --log logs/qwen3-235b-a22b-bioASQ-gpt-generation.log

python src/gpt_test/bioASQ_gpt_api.py --answer data/evaluation/model_answer/DeepSeek/deepseek-r1-distill-qwen-1.5b/deepseek-r1-distill-qwen-1.5b-bioASQ-answer.json --output data/evaluation/gpt_test/deepseek-r1-distill-qwen-1.5b-bioASQ-gpt.json --log logs/deepseek-r1-distill-qwen-1.5b-bioASQ-gpt-generation.log

python src/gpt_test/bioASQ_gpt_api.py --answer data/evaluation/model_answer/DeepSeek/deepseek-r1-distill-llama-8b/deepseek-r1-distill-llama-8b-bioASQ-answer.json --output data/evaluation/gpt_test/deepseek-r1-distill-llama-8b-bioASQ-gpt.json --log logs/deepseek-r1-distill-llama-8b-bioASQ-gpt-generation.log

python src/gpt_test/bioASQ_gpt_api.py --answer data/evaluation/model_answer/DeepSeek/deepseek-r1-distill-qwen-14b/deepseek-r1-distill-qwen-14b-bioASQ-answer.json --output data/evaluation/gpt_test/deepseek-r1-distill-qwen-14b-bioASQ-gpt.json --log logs/deepseek-r1-distill-qwen-14b-bioASQ-gpt-generation.log

python src/gpt_test/bioASQ_gpt_api.py --answer data/evaluation/model_answer/DeepSeek/deepseek-r1-distill-qwen-32b/deepseek-r1-distill-qwen-32b-bioASQ-answer.json --output data/evaluation/gpt_test/deepseek-r1-distill-qwen-32b-bioASQ-gpt.json --log logs/deepseek-r1-distill-qwen-32b-bioASQ-gpt-generation.log

python src/gpt_test/bioASQ_gpt_api.py --answer data/evaluation/model_answer/DeepSeek/deepseek-r1-distill-llama-70b/deepseek-r1-distill-llama-70b-bioASQ-answer.json --output data/evaluation/gpt_test/deepseek-r1-distill-llama-70b-bioASQ-gpt.json --log logs/deepseek-r1-distill-llama-70b-bioASQ-gpt-generation.log

python src/gpt_test/bioASQ_gpt_api.py --answer data/evaluation/model_answer/DeepSeek/deepseek-r1/deepseek-r1-bioASQ-answer.json --output data/evaluation/gpt_test/deepseek-r1-bioASQ-gpt.json --log logs/deepseek-r1-bioASQ-gpt-generation.log

python src/gpt_test/chemprot_gpt_api.py --answer data/evaluation/model_answer/Qwen/qwen3-4b/qwen3-4b-chemprot-answer.json --output data/evaluation/gpt_test/qwen3-4b-chemprot-gpt.json --log logs/qwen3-4b-chemprot-gpt-generation.log

python src/gpt_test/chemprot_gpt_api.py --answer data/evaluation/model_answer/Qwen/qwen3-8b/qwen3-8b-chemprot-answer.json --output data/evaluation/gpt_test/qwen3-8b-chemprot-gpt.json --log logs/qwen3-8b-chemprot-gpt-generation.log

python src/gpt_test/chemprot_gpt_api.py --answer data/evaluation/model_answer/Qwen/qwen3-14b/qwen3-14b-chemprot-answer.json --output data/evaluation/gpt_test/qwen3-14b-chemprot-gpt.json --log logs/qwen3-14b-chemprot-gpt-generation.log

python src/gpt_test/chemprot_gpt_api.py --answer data/evaluation/model_answer/Qwen/qwen3-30b-a3b/qwen3-30b-a3b-chemprot-answer.json --output data/evaluation/gpt_test/qwen3-30b-a3b-chemprot-gpt.json --log logs/qwen3-30b-a3b-chemprot-gpt-generation.log

python src/gpt_test/chemprot_gpt_api.py --answer data/evaluation/model_answer/Qwen/qwen3-32b/qwen3-32b-chemprot-answer.json --output data/evaluation/gpt_test/qwen3-32b-chemprot-gpt.json --log logs/qwen3-32b-chemprot-gpt-generation.log

python src/gpt_test/chemprot_gpt_api.py --answer data/evaluation/model_answer/Qwen/qwen3-235b-a22b/qwen3-235b-a22b-chemprot-answer.json --output data/evaluation/gpt_test/qwen3-235b-a22b-chemprot-gpt.json --log logs/qwen3-235b-a22b-chemprot-gpt-generation.log

python src/gpt_test/chemprot_gpt_api.py --answer data/evaluation/model_answer/DeepSeek/deepseek-r1-distill-qwen-1.5b/deepseek-r1-distill-qwen-1.5b-chemprot-answer.json --output data/evaluation/gpt_test/deepseek-r1-distill-qwen-1.5b-chemprot-gpt.json --log logs/deepseek-r1-distill-qwen-1.5b-chemprot-gpt-generation.log

python src/gpt_test/chemprot_gpt_api.py --answer data/evaluation/model_answer/DeepSeek/deepseek-r1-distill-llama-8b/deepseek-r1-distill-llama-8b-chemprot-answer.json --output data/evaluation/gpt_test/deepseek-r1-distill-llama-8b-chemprot-gpt.json --log logs/deepseek-r1-distill-llama-8b-chemprot-gpt-generation.log

python src/gpt_test/chemprot_gpt_api.py --answer data/evaluation/model_answer/DeepSeek/deepseek-r1-distill-qwen-14b/deepseek-r1-distill-qwen-14b-chemprot-answer.json --output data/evaluation/gpt_test/deepseek-r1-distill-qwen-14b-chemprot-gpt.json --log logs/deepseek-r1-distill-qwen-14b-chemprot-gpt-generation.log

python src/gpt_test/chemprot_gpt_api.py --answer data/evaluation/model_answer/DeepSeek/deepseek-r1-distill-qwen-32b/deepseek-r1-distill-qwen-32b-chemprot-answer.json --output data/evaluation/gpt_test/deepseek-r1-distill-qwen-32b-chemprot-gpt.json --log logs/deepseek-r1-distill-qwen-32b-chemprot-gpt-generation.log

python src/gpt_test/chemprot_gpt_api.py --answer data/evaluation/model_answer/DeepSeek/deepseek-r1-distill-llama-70b/deepseek-r1-distill-llama-70b-chemprot-answer.json --output data/evaluation/gpt_test/deepseek-r1-distill-llama-70b-chemprot-gpt.json --log logs/deepseek-r1-distill-llama-70b-chemprot-gpt-generation.log

python src/gpt_test/chemprot_gpt_api.py --answer data/evaluation/model_answer/DeepSeek/deepseek-r1/deepseek-r1-chemprot-answer.json --output data/evaluation/gpt_test/deepseek-r1-chemprot-gpt.json --log logs/deepseek-r1-chemprot-gpt-generation.log
