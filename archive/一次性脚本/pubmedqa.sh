#!/bin/bash
set -e  # 遇到错误立即退出

#python src/gpt_test/pubmedqa_gpt_api.py --answer data/evaluation/model_answer/qwen3-8b-pubmedqa-answer-generation.json --output data/evaluation/gpt_test/qwen3-8b-pubmedqa-gpt.json --log logs/qwen3-8b-pubmedqa-gpt.log
#python src/gpt_test/pubmedqa_gpt_api.py --answer data/evaluation/model_answer/Qwen3-8B-sft-pubmedqa-answer.json --output data/evaluation/gpt_test/qwen3-8b-sft-pubmedqa-gpt.json --log logs/qwen3-8b-sft-pubmedqa-gpt.log
python src/gpt_test/pubmedqa_gpt_api.py --answer data/evaluation/model_answer/ckpt-3000-kto-ckpt-3500-pubmedqa-answer.json --output data/evaluation/gpt_test/ckpt-3000-kto-ckpt-3500-pubmedqa-gpt.json --log logs/ckpt-3000-kto-ckpt-3500-pubmedqa-gpt.log

python src/gpt_result/gpt_result.py --test data/evaluation/gpt_test/qwen3-8b-pubmedqa-gpt.json --result data/evaluation/gpt_result/qwen3-8b-pubmedqa-gpt.txt
python src/gpt_result/gpt_result.py --test data/evaluation/gpt_test/qwen3-8b-sft-pubmedqa-gpt.json --result data/evaluation/gpt_result/qwen3-8b-sft-pubmedqa-gpt.txt
python src/gpt_result/gpt_result.py --test data/evaluation/gpt_test/ckpt-3000-kto-ckpt-3500-pubmedqa-gpt.json --result data/evaluation/gpt_result/ckpt-3000-kto-ckpt-3500-pubmedqa-gpt.txt
