#!/bin/bash
set -e  # 遇到错误立即退出

python src/model_answer_generation/pharmacy_answer.py --model_dir model/Qwen3-8B-sft --output data/evaluation/model_answer/qwen3-8b-sft-pharmacy-answer.json --device cuda --log logs/qwen3-8b-sft-pharmacy-answer.log --enable_thinking
python src/model_answer_generation/pharmacy_answer.py --model_dir model/Qwen3-8B-kto --output data/evaluation/model_answer/qwen3-8b-kto-pharmacy-answer.json --device cuda --log logs/qwen3-8b-kto-pharmacy-answer.log --enable_thinking