import json
from transformers import AutoTokenizer

# 配置你的本地模型路径或HuggingFace模型名
tokenizer = AutoTokenizer.from_pretrained("./model/Qwen3-8B-sft/checkpoint-3000")

# 指定你的json文件路径
json_file = "./data/grpo_train.json"

# 加载数据
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 统计每条output的token长度
lengths = [len(tokenizer.encode(item["output"])) for item in data if "output" in item]

print(f"最大 token 长度: {max(lengths)}")
print(f"最小 token 长度: {min(lengths)}")
print(f"平均 token 长度: {sum(lengths) / len(lengths):.2f}")
print(f"样本数: {len(lengths)}")