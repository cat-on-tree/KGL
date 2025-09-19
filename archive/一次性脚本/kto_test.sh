#!/bin/bash
set -e  # 遇到错误立即退出
#三个test脚本
#python src/model_test/biored_test_api.py --input data/evaluation/model_answer/ckpt-3000-kto-ckpt-4000-bioRED-answer.json --output data/evaluation/model_test/ckpt-3000-kto-ckpt-4000-bioRED-test.json --log logs/ckpt-3000-kto-ckpt-4000-bioRED-test.log
#python src/model_test/chemprot_test_api.py --input data/evaluation/model_answer/ckpt-3000-kto-ckpt-4000-chemprot-answer.json --output data/evaluation/model_test/ckpt-3000-kto-ckpt-4000-chemprot-test.json --log logs/ckpt-3000-kto-ckpt-4000-chemprot-test.log
#python src/model_test/bioASQ_test_api.py --input data/evaluation/model_answer/ckpt-3000-kto-ckpt-4000-bioASQ-answer.json --output data/evaluation/model_test/ckpt-3000-kto-ckpt-4000-bioASQ-test.json --log logs/ckpt-3000-kto-ckpt-4000-bioASQ-test.log
##四个result脚本
#python src/model_result/biored_test_result.py --test data/evaluation/model_test/ckpt-3000-kto-ckpt-4000-bioRED-test.json --result data/evaluation/model_result/ckpt-3000-kto-ckpt-4000-bioRED-result.txt
#python src/model_result/chemprot_test_result.py --test data/evaluation/model_test/ckpt-3000-kto-ckpt-4000-chemprot-test.json --result data/evaluation/model_result/ckpt-3000-kto-ckpt-4000-chemprot-result.txt
#python src/model_result/bioASQ_test_result.py --test data/evaluation/model_test/ckpt-3000-kto-ckpt-4000-bioASQ-test.json --result data/evaluation/model_result/ckpt-3000-kto-ckpt-4000-bioASQ-result.txt
#python src/model_result/bioASQ_bert_result.py --answer data/evaluation/model_answer/ckpt-3000-kto-ckpt-4000-bioASQ-answer.json --result data/evaluation/model_result/ckpt-3000-kto-ckpt-4000-bioASQ-bert.txt --log logs/ckpt-3000-kto-ckpt-4000-bioASQ-bert.log
#三个gpt_score生成脚本
#python src/gpt_test/biored_gpt_api.py --answer data/evaluation/model_answer/ckpt-3000-kto-ckpt-4000-bioRED-answer.json --output data/evaluation/gpt_test/ckpt-3000-kto-ckpt-4000-bioRED-gpt.json --log logs/ckpt-3000-kto-ckpt-4000-bioRED-gpt.log
#python src/gpt_test/chemprot_gpt_api.py --answer data/evaluation/model_answer/ckpt-3000-kto-ckpt-4000-chemprot-answer.json --output data/evaluation/gpt_test/ckpt-3000-kto-ckpt-4000-chemprot-gpt.json --log logs/ckpt-3000-kto-ckpt-4000-chemprot-gpt.log
python src/gpt_test/bioASQ_gpt_api.py --answer data/evaluation/model_answer/ckpt-3000-kto-ckpt-4000-bioASQ-answer.json --output data/evaluation/gpt_test/ckpt-3000-kto-ckpt-4000-bioASQ-gpt.json --log logs/ckpt-3000-kto-ckpt-4000-bioASQ-gpt.log
#三个gpt_score结果脚本
python src/gpt_result/gpt_result.py --test data/evaluation/gpt_test/ckpt-3000-kto-ckpt-4000-bioRED-gpt.json --result data/evaluation/gpt_result/ckpt-3000-kto-ckpt-4000-bioRED-gpt.txt
python src/gpt_result/gpt_result.py --test data/evaluation/gpt_test/ckpt-3000-kto-ckpt-4000-chemprot-gpt.json --result data/evaluation/gpt_result/ckpt-3000-kto-ckpt-4000-chemprot-gpt.txt
python src/gpt_result/gpt_result.py --test data/evaluation/gpt_test/ckpt-3000-kto-ckpt-4000-bioASQ-gpt.json --result data/evaluation/gpt_result/ckpt-3000-kto-ckpt-4000-bioASQ-gpt.txt
