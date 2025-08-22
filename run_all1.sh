#!/bin/zsh
set -e  # 遇到错误立即退出
#三个生成脚本
python src/model_answer_generation/biored_answer_api.py --model deepseek-r1-distill-llama-8b --output data/evaluation/model_answer/deepseek-r1-distill-llama-8b-biored-answer.json --threads 2 --log logs/deepseek-r1-distill-llama-8b-biored-answer-generation.log

python src/model_answer_generation/chemprot_answer_api.py --model deepseek-r1-distill-llama-8b --output data/evaluation/model_answer/deepseek-r1-distill-llama-8b-chemprot-answer.json --threads 2 --log logs/deepseek-r1-distill-llama-8b-chemprot-answer-generation.log

python src/model_answer_generation/bioasq_answer_api.py --model deepseek-r1-distill-llama-8b --output data/evaluation/model_answer/deepseek-r1-distill-llama-8b-bioASQ-answer.json --threads 2 --log logs/deepseek-r1-distill-llama-8b-bioASQ-answer-generation.log
#三个评测脚本
python src/model_performance_summary/biored_test_api.py --input data/evaluation/model_answer/deepseek-r1-distill-llama-8b-biored-answer.json --output data/evaluation/model_test/deepseek-r1-distill-llama-8b-biored-test.json --log logs/deepseek-r1-distill-llama-8b-biored-model-test.log

python src/model_performance_summary/chemprot_test_api.py --input data/evaluation/model_answer/deepseek-r1-distill-llama-8b-chemprot-answer.json --output data/evaluation/model_test/deepseek-r1-distill-llama-8b-chemprot-test.json --log logs/deepseek-r1-distill-llama-8b-chemprot-model-test.log

python src/model_performance_summary/bioASQ_test_api.py --input data/evaluation/model_answer/deepseek-r1-distill-llama-8b-bioASQ-answer.json --output data/evaluation/model_test/deepseek-r1-distill-llama-8b-bioASQ-test.json --log logs/deepseek-r1-distill-llama-8b-bioASQ-model-test.log
#四个结果脚本
python src/model_test/biored_test_result.py --test data/evaluation/model_test/deepseek-r1-distill-llama-8b-biored-test.json --result data/evaluation/model_result/deepseek-r1-distill-llama-8b-biored-result.txt

python src/model_test/chemprot_test_result.py --test data/evaluation/model_test/deepseek-r1-distill-llama-8b-chemprot-test.json --result data/evaluation/model_result/deepseek-r1-distill-llama-8b-chemprot-result.txt

python src/model_test/bioASQ_test_result.py --test data/evaluation/model_test/deepseek-r1-distill-llama-8b-bioASQ-test.json --result data/evaluation/model_result/deepseek-r1-distill-llama-8b-bioASQ-result.txt

python src/model_test/bioASQ_bert_result.py --answer data/evaluation/model_answer/deepseek-r1-distill-llama-8b-bioASQ-answer.json --result data/evaluation/model_result/deepseek-r1-distill-llama-8b-bioASQ-bert.txt --log logs/deepseek-r1-distill-llama-8b-bioASQ-bert.log
