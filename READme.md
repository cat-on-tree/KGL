model answer generation 中的脚本用法：

```apache
python xx.py --model xx --output xx.json --threads 4 --log logs/ #enable_thinking (Qwen系列开源模型)
```

对于本地模型推理：

```apache
python xx.py --model_dir model/自己的模型 --output xx.json --device mps/cuda/cpu --log logs/
```

model test中的脚本用法：

```apache
python src/model_performance_summary/xx_test_api.py --input data/evaluation/model_answer/xx-answer.json --output data/evaluation/model_test/xx-test.json --log logs/

```

model result中的脚本用法：

```apache
python src/model_test/xx.py --test data/evaluation/model_test/xx-test.json --result data/evaluation/model_result/xx-result.txt
```

对于model result中的bioASQ BERT指标计算：

```apache
python src/model_test/bioASQ_bert_result.py --answer data/evaluation/model_answer/xx-answer.json --result data/evaluation/model_result/xx-bert.txt --log logs/
```

之后是计算gpt score：

```apache
python src/gpt_test/xx_gpt_api.py --answer data/evaluation/model_answer/xx-answer.json --result data/evaluation/gpt_test/model_data-gpt.json --log logs/
```
