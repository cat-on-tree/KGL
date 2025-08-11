import json
import re
import os
import time
import random
from openai import OpenAI
from tqdm import tqdm

def parse_entry(entry):
    """
    解析单条数据，返回question, answer, context_sentences
    """
    question = entry['question']
    text = entry['text']
    answer_match = re.search(r'<answer>\s*(.*?)\s*<context>', text, re.DOTALL)
    context_match = re.search(r'<context>\s*(.*)', text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""
    context = context_match.group(1).strip() if context_match else ""
    context_sentences = re.split(r'(?<=[.!?])\s+', context)
    context_sentences = [s for s in context_sentences if s.strip()]
    return {
        'question': question,
        'answer': answer,
        'context_sentences': context_sentences
    }

def call_llm_get_supporting_sentences(question, answer, context_sentences, client, model="qwen-max-latest", max_retry=3):
    """
    调用大语言模型，获取supporting_sentences
    """
    example1 = {
        "question": "What is the inheritance pattern of Li–Fraumeni syndrome?",
        "answer": "autosomal dominant",
        "context_sentences": [
            "Li-Fraumeni Syndrome (LFS) is characterized by early-onset carcinogenesis involving multiple tumor types and shows autosomal dominant inheritance.",
            "Mutations in the TP53 gene are frequently observed in LFS families."
        ]
    }
    example1_response = {
        "supporting_sentences": [
            "Li-Fraumeni Syndrome (LFS) is characterized by early-onset carcinogenesis involving multiple tumor types and shows autosomal dominant inheritance."
        ]
    }

    example2 = {
        "question": "Which hormone abnormalities are characteristic to Pendred syndrome?",
        "answer": "thyroid",
        "context_sentences": [
            "Loss or reduction of function mutations of SLC26A4 underlie Pendred syndrome, a disorder invariably leading to hearing loss with enlarged vestibular aqueducts and in some patients to hypothyroidism and goiter.",
            "Pendrin is expressed in inner ear, thyroid gland, kidneys, lung, liver and heart."
        ]
    }
    example2_response = {
        "supporting_sentences": [
            "Loss or reduction of function mutations of SLC26A4 underlie Pendred syndrome, a disorder invariably leading to hearing loss with enlarged vestibular aqueducts and in some patients to hypothyroidism and goiter."
        ]
    }

    system_prompt = f"""你是医学推理助手。请根据用户输入的“问题”、“标准答案”和“上下文句子列表”，找出直接支撑标准答案的句子，并以JSON字符串输出。输出格式如下：supporting_sentences（数组类型，内容为原文句子）。

示例：
Q：问题：{example1["question"]} 答案：{example1["answer"]} 上下文：{example1["context_sentences"]}
A：{json.dumps(example1_response, ensure_ascii=False)}

Q：问题：{example2["question"]} 答案：{example2["answer"]} 上下文：{example2["context_sentences"]}
A：{json.dumps(example2_response, ensure_ascii=False)}
"""

    user_prompt = f'问题：{question} 答案：{answer} 上下文：{context_sentences}'

    retry = 0
    while retry < max_retry:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
            )
            json_string = completion.choices[0].message.content
            response = json.loads(json_string)
            return response.get("supporting_sentences", [])
        except Exception as e:
            print(f"调用模型出错，重试中... ({retry+1}/{max_retry}) 错误信息: {e}")
            time.sleep(2)
            retry += 1
    return []

def move_idx_first(d: dict) -> dict:
    """
    将字典中的'idx'字段移到最前面
    """
    if "idx" not in d:
        return d
    items = list(d.items())
    # 找到idx的位置并弹出
    idx_pair = None
    for i, pair in enumerate(items):
        if pair[0] == "idx":
            idx_pair = items.pop(i)
            break
    if idx_pair:
        items = [idx_pair] + items
    return dict(items)

def process_and_merge_sampled(input_file, output_file, api_key, sample_num=1000, seed=42, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"):
    random.seed(seed)
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    # 读取全部原始数据
    all_entries = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_entries.append(json.loads(line))
    # 随机采样（有种子保证复现性）
    if len(all_entries) < sample_num:
        print(f"警告：原始数据仅有{len(all_entries)}条，少于{sample_num}，将全部处理。")
        sampled_entries = all_entries
    else:
        sampled_entries = random.sample(all_entries, sample_num)

    parsed_entries = [parse_entry(entry) for entry in sampled_entries]

    merged_results = []
    for idx, (raw_entry, parsed_entry) in enumerate(tqdm(zip(sampled_entries, parsed_entries), total=len(parsed_entries), desc="Processing Sampled Data")):
        supporting_sentences = call_llm_get_supporting_sentences(
            parsed_entry['question'],
            parsed_entry['answer'],
            parsed_entry['context_sentences'],
            client
        )
        merged = raw_entry.copy()
        merged['supporting_sentences'] = supporting_sentences
        merged['idx'] = idx  # 从0开始编号
        merged = move_idx_first(merged)
        merged_results.append(merged)
        # 实时保存，防止中断丢失
        with open(output_file, 'w', encoding='utf-8') as wf:
            for item in merged_results:
                wf.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量DASHSCOPE_API_KEY")
    input_file = "BioASQ_test.json"            # 原始输入文件
    output_file = "../../data/evaluation/benchmark/BioASQ.json"  # 合并后输出文件
    process_and_merge_sampled(input_file, output_file, api_key, sample_num=1000, seed=42)