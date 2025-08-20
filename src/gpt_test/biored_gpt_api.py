import os
import json
from openai import OpenAI
from tqdm import tqdm
import argparse
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def extract_user_and_label(data_text):
    """
    从data字段中提取user prompt和金标准标签
    """
    sys_tag = "<</SYS>>\n"
    inst_tag = "[/INST]"
    sys_pos = data_text.find(sys_tag)
    inst_pos = data_text.find(inst_tag)
    if sys_pos == -1 or inst_pos == -1:
        return "", ""
    user_prompt = data_text[sys_pos + len(sys_tag):inst_pos].strip()
    gold_label = data_text[inst_pos + len(inst_tag):].strip()
    gold_label = re.sub(r"<.*?>", "", gold_label).strip()
    return user_prompt, gold_label

def build_prompt(user_prompt, model_answer, gold_label):
    return (
        f"Question:\n{user_prompt}\n\n"
        f"Model Answer:\n{model_answer}\n\n"
        f"Gold Label:\n{gold_label}\n\n"
        f"Please rate the answer and explain your score in JSON format as instructed."
    )

def score_one(idx, user_prompt, model_answer, gold_label, system_prompt, client, model, max_retries=5):
    user_prompt_full = build_prompt(user_prompt, model_answer, gold_label)
    wait_times = [5, 10, 10, 10, 10]  # 总共5次，第一次失败后5s，后面4次10s
    last_exception = None
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_full},
                ],
                response_format={"type": "json_object"},
            )
            result = completion.choices[0].message.content
            # 检查返回内容是否为合法JSON
            try:
                json.loads(result)
            except Exception as e:
                logging.error(f"idx={idx} LLM输出不是合法JSON: {e}")
                return {
                    "idx": idx,
                    "error": f"LLM输出不是合法JSON: {e}",
                    "gold_label": gold_label,
                    "model_answer": model_answer,
                    "retries": attempt+1
                }
            return {
                "idx": idx,
                "gptscore_json": result,
                "gold_label": gold_label,
                "model_answer": model_answer,
                "retries": attempt+1
            }
        except Exception as e:
            last_exception = e
            logging.warning(f"idx={idx} 第{attempt+1}次请求出错: {e}")
            if attempt < max_retries - 1:
                time.sleep(wait_times[attempt])
    # 5次都失败
    errmsg = f"idx={idx} 请求重试{max_retries}次仍失败: {last_exception}"
    logging.error(errmsg)
    return {
        "idx": idx,
        "error": errmsg,
        "gold_label": gold_label,
        "model_answer": model_answer,
        "retries": max_retries
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-max-latest", help="评价用的大模型名称")
    parser.add_argument("--benchmark", default="data/evaluation/benchmark/bioRED.json", help="输入benchmark.json文件名")
    parser.add_argument("--answer", required=True, help="待评价答案的jsonl文件名")
    parser.add_argument("--result", required=True, help="输出jsonl文件名")
    parser.add_argument("--log", default="biored_gptscore_test.log", help="日志文件名")
    parser.add_argument("--api_key", default=os.getenv("DASHSCOPE_API_KEY"), help="API KEY")
    parser.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="API base url")
    parser.add_argument("--threads", type=int, default=4, help="并发线程数量")
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # few-shot 示例，包含answer字段，但LLM只需输出score/reason/match
    example_5 = json.dumps({
        "answer": "Based on the provided abstract and title, the SCN5A mutation results in bradycardia during perinatal onset. The evidence shows a persistent inward current due to the mutation, which is associated with arrhythmias including bradycardia. The text does not explicitly claim causality, so the most accurate relationship is 'Association'.",
        "score": 5,
        "reason": "Label matches gold standard. The answer provides detailed and logically sound reasoning, referencing specific evidence from the user_prompt. The explanation is comprehensive and clear.",
        "match": True
    }, ensure_ascii=False)
    example_4 = json.dumps({
        "answer": "SCN5A mutation is associated with bradycardia. Relationship type: Association.",
        "score": 4,
        "reason": "Label matches gold standard. Reasoning is generally reasonable but lacks depth and does not cite specific evidence from the user_prompt.",
        "match": True
    }, ensure_ascii=False)
    example_3 = json.dumps({
        "answer": "Association.",
        "score": 3,
        "reason": "Label matches gold standard, but the answer lacks any reasoning or reference to the user_prompt.",
        "match": True
    }, ensure_ascii=False)
    example_2 = json.dumps({
        "answer": "SCN5A mutation causes bradycardia. Relationship type: Positive_Correlation.",
        "score": 2,
        "reason": "Label does not match gold standard, but the reasoning is internally consistent and relates to the user_prompt, showing some understanding of the text.",
        "match": False
    }, ensure_ascii=False)
    example_1 = json.dumps({
        "answer": "No relationship is described.",
        "score": 1,
        "reason": "Label does not match gold standard and the answer lacks any reasonable reference to the user_prompt or contains irrelevant content.",
        "match": False
    }, ensure_ascii=False)

    system_prompt = f"""
You are an expert biomedical information extractor.
Given a question (user_prompt), a model's answer,
and a gold standard label, rate the model's answer according to the following criteria:

Scoring reference:
- 5: Label matches gold standard, reasoning is thorough and logically sound, references key evidence from user_prompt, content is rich and explanation is clear.
- 4: Label matches gold standard, reasoning is generally reasonable but lacks detail or depth, some logical flaws or missing evidence, content is somewhat brief.
- 3: Label matches gold standard, but reasoning is minimal or absent, only direct label or simple repetition, lacks analytical process.
- 2: Label does not match gold standard, but reasoning is internally consistent and relates to user_prompt, showing some understanding.
- 1: Label does not match gold standard, and answer lacks any reasonable reference to user_prompt or is irrelevant/meaningless.

Output JSON fields:
- "score": integer, 1 (very poor) to 5 (excellent)
- "reason": string, explaining your score, especially the reasoning evaluation
- "match": boolean, true if the answer matches the gold label, false otherwise

Do NOT include the 'answer' field in your output JSON.  
The examples provided below include 'answer' for illustration only.

Output format examples:
{example_5}
{example_4}
{example_3}
{example_2}
{example_1}

Please only output the JSON object with the specified fields.
    """

    # 读取 benchmark.json
    with open(args.benchmark, "r", encoding="utf-8") as fin:
        benchmark_items = [json.loads(line) for line in fin if line.strip()]
    idx2benchmark = {item["idx"]: item for item in benchmark_items}

    # 读取答案 jsonl
    with open(args.answer, "r", encoding="utf-8") as fin:
        answer_items = [json.loads(line) for line in fin if line.strip()]
    idx2answer = {item["idx"]: item for item in answer_items}

    # 构造所有任务
    tasks = []
    for cur_idx in sorted(idx2benchmark.keys()):
        benchmark_obj = idx2benchmark[cur_idx]
        answer_obj = idx2answer.get(cur_idx, None)
        data_text = benchmark_obj.get("data", "")
        user_prompt, gold_label = extract_user_and_label(data_text)
        model_answer = answer_obj.get("llm_output", "").strip() if answer_obj else ""
        tasks.append((cur_idx, user_prompt, model_answer, gold_label))

    results = {}
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        future2idx = {
            executor.submit(
                score_one, idx, user_prompt, model_answer, gold_label, system_prompt,
                client, args.model, 5
            ): idx
            for idx, user_prompt, model_answer, gold_label in tasks
        }
        for future in tqdm(as_completed(future2idx), total=len(future2idx), desc="Concurrent LLM scoring w/ retry"):
            idx = future2idx[future]
            try:
                out = future.result()
            except Exception as e:
                out = {
                    "idx": idx,
                    "error": f"Threaded error: {e}",
                    "gold_label": "",
                    "model_answer": "",
                    "retries": 5
                }
            results[idx] = out

    # 按照输入顺序写出结果，保证idx一致性
    with open(args.result, "w", encoding="utf-8") as fout:
        for cur_idx in sorted(idx2benchmark.keys()):
            out = results.get(cur_idx, {"idx": cur_idx, "error": "No result generated.", "retries": 5})
            out["idx"] = cur_idx  # 显式确保idx一致性
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
    logging.info(f"全部评分处理完成，结果保存在 {args.result}")

if __name__ == "__main__":
    main()