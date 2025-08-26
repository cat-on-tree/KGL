#!/bin/bash
set -e  # 遇到错误立即退出

#三个评测脚本
python src/model_test/biored_test_api.py --input data/evaluation/model_answer/Qwen3-8b-sft-checkpoint-4500-bioRED-answer.json --output data/evaluation/model_test/Qwen3-8b-sft-checkpoint-4500-biored-test.json --log logs/Qwen3-8b-sft-checkpoint-4500-biored-model-test.log

python src/model_test/chemprot_test_api.py --input data/evaluation/model_answer/Qwen3-8b-sft-checkpoint-4500-chemprot-answer.json --output data/evaluation/model_test/Qwen3-8b-sft-checkpoint-4500-chemprot-test.json --log logs/Qwen3-8b-sft-checkpoint-4500-chemprot-model-test.log

python src/model_test/bioASQ_test_api.py --input data/evaluation/model_answer/Qwen3-8b-sft-checkpoint-4500-bioASQ-answer.json --output data/evaluation/model_test/Qwen3-8b-sft-checkpoint-4500-bioASQ-test.json --log logs/Qwen3-8b-sft-checkpoint-4500-bioASQ-model-test.log
#四个结果脚本
python src/model_result/biored_test_result.py --test data/evaluation/model_test/Qwen3-8b-sft-checkpoint-4500-biored-test.json --result data/evaluation/model_result/Qwen3-8b-sft-checkpoint-4500-biored-result.txt

python src/model_result/chemprot_test_result.py --test data/evaluation/model_test/Qwen3-8b-sft-checkpoint-4500-chemprot-test.json --result data/evaluation/model_result/Qwen3-8b-sft-checkpoint-4500-chemprot-result.txt

python src/model_result/bioASQ_test_result.py --test data/evaluation/model_test/Qwen3-8b-sft-checkpoint-4500-bioASQ-test.json --result data/evaluation/model_result/Qwen3-8b-sft-checkpoint-4500-bioASQ-result.txt

python src/model_result/bioASQ_bert_result.py --answer data/evaluation/model_answer/Qwen3-8b-sft-checkpoint-4500-bioASQ-answer.json --result data/evaluation/model_result/Qwen3-8b-sft-checkpoint-4500-bioASQ-bert.txt --log logs/Qwen3-8b-sft-checkpoint-4500-bioASQ-bert.log

#三个gpt评测脚本

python src/gpt_test/biored_gpt_api.py --answer data/evaluation/model_answer/Qwen3-8b-sft-checkpoint-4500-bioRED-answer.json --output data/evaluation/gpt_test/Qwen3-8b-sft-checkpoint-4500-biored-gpt.json --log logs/Qwen3-8b-sft-checkpoint-4500-biored-gpt-generation.log

python src/gpt_test/chemprot_gpt_api.py --answer data/evaluation/model_answer/Qwen3-8b-sft-checkpoint-4500-chemprot-answer.json --output data/evaluation/gpt_test/Qwen3-8b-sft-checkpoint-4500-chemprot-gpt.json --log logs/Qwen3-8b-sft-checkpoint-4500-chemprot-gpt-generation.log

python src/gpt_test/bioASQ_gpt_api.py --answer data/evaluation/model_answer/Qwen3-8b-sft-checkpoint-4500-bioASQ-answer.json --output data/evaluation/gpt_test/Qwen3-8b-sft-checkpoint-4500-bioASQ-gpt.json --log logs/Qwen3-8b-sft-checkpoint-4500-bioASQ-gpt-generation.log

#三个gpt结果脚本

python src/gpt_result/gpt_result.py --test data/evaluation/gpt_test/Qwen3-8b-sft-checkpoint-4500-biored-gpt.json --result data/evaluation/gpt_result/Qwen3-8b-sft-checkpoint-4500-biored-gpt.txt

python src/gpt_result/gpt_result.py --test data/evaluation/gpt_test/Qwen3-8b-sft-checkpoint-4500-chemprot-gpt.json --result data/evaluation/gpt_result/Qwen3-8b-sft-checkpoint-4500-chemprot-gpt.txt

python src/gpt_result/gpt_result.py --test data/evaluation/gpt_test/Qwen3-8b-sft-checkpoint-4500-bioASQ-gpt.json --result data/evaluation/gpt_result/Qwen3-8b-sft-checkpoint-4500-bioASQ-gpt.txt

