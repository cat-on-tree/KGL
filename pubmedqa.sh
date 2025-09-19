#!/bin/bash
set -e  # 遇到错误立即退出

#python src/model_answer_generation/pubmedqa_answer.py --model_dir model/Qwen3-8B-kto/checkpoint-3500 --output data/evaluation/model_answer/ckpt-3000-kto-ckpt-3500-pubmedqa-answer.json --log logs/ckpt-3000-kto-ckpt-3500-pubmedqa-answer-generation.log --enable_thinking True --device cuda
python src/model_answer_generation/pubmedqa_answer_api.py --model qwen3-8b --output data/evaluation/model_answer/qwen3-8b-pubmedqa-answer.json --log logs/qwen3-8b-pubmedqa-answer-generation.log --enable_thinking True