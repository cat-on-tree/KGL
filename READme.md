model answer generation 中的脚本用法：

```apache
python xx.py --model xx --output xx.json --threads 4 --log logs/xx-answer.log #enable_thinking (Qwen系列开源模型)
```

对于本地模型推理：

```apache
python xx.py --model_dir model/自己的模型 --output xx.json --device mps/cuda/cpu --log logs/xx-answer.log
```

超参数分析：

```apache
python xx.py --model_dir model/自己的模型 --output xx.json --device mps/cuda/cpu --log logs/xx-answer.log --temperature (取值[0,2)) --top_k (取值[0,100))
```

model test中的脚本用法：

```apache
python src/model_test/xx_test_api.py --input data/evaluation/model_answer/xx-answer.json --output data/evaluation/model_test/xx-test.json --log logs/xx-test.log

```

model result中的脚本用法：

```apache
python src/model_result/xx.py --test data/evaluation/model_test/xx-test.json --result data/evaluation/model_result/xx-result.txt
```

对于model result中的BERT指标计算：

```apache
python src/model_test/xx_bert_result.py --answer data/evaluation/model_answer/xx-answer.json --result data/evaluation/model_result/xx-bert.txt --log logs/xx-bert.log
```

之后是计算gpt score：

```apache
python src/gpt_test/xx_gpt_api.py --answer data/evaluation/model_answer/xx-answer.json --output data/evaluation/gpt_test/xx-gpt.json --log logs/xx-gpt.log
```

统计gpt score结果：

```apache
python src/gpt_result/gpt_result.py --test data/evaluation/gpt_test/xx-gpt.json --result data/evaluation/gpt_result/xx-gpt.txt
```
