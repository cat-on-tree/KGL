import os
import json
from openai import OpenAI
from tqdm import tqdm
import argparse
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def extract_task_input_gold(item):
    """
    从benchmark每条item中提取task描述、输入句子、金标准标签
    """
    # 通用任务、输入、标准
    task_desc = "the task is to classify relations between a chemical and a gene for a sentence."
    input_desc = "the input is a sentence where the chemical is labeled as @CHEMICAL$ and the gene is labeled as @GENE$ accordingly in a sentence."
    output_desc = ("your task is to select one out of the six types of relations "
                   "('CPR:3', 'CPR:4', 'CPR:5', 'CPR:6', 'CPR:9', and 'false') for the gene and chemical.")
    # 每条输入句子
    sentence = ""
    gold_label = ""
    if "unprocessed" in item and "processed" in item:
        # 提取input部分（最后一问），和金标准
        unprocessed = item["unprocessed"]
        # 提取最后一个Q: ...（即真正input）
        sentence_match = re.search(r'Q:\s*(.*)$', unprocessed.strip(), re.MULTILINE)
        if sentence_match:
            sentence = sentence_match.group(1).strip()
        gold_label = item["processed"].strip()
    else:
        sentence = ""
        gold_label = ""
    return task_desc, input_desc, output_desc, sentence, gold_label

def build_prompt(task_desc, input_desc, output_desc, sentence, model_answer, gold_label):
    return (
        f"TASK:\n{task_desc}\n\n"
        f"INPUT:\n{input_desc}\n\n"
        f"OUTPUT:\n{output_desc}\n\n"
        f"Sentence:\n{sentence}\n\n"
        f"Model Answer:\n{model_answer}\n\n"
        f"Gold Label:\n{gold_label}\n\n"
        f"Please rate the answer and explain your score in JSON format as instructed."
    )

def score_one(idx, task_desc, input_desc, output_desc, sentence, model_answer, gold_label, system_prompt, client, model, max_retries=5):
    user_prompt_full = build_prompt(task_desc, input_desc, output_desc, sentence, model_answer, gold_label)
    wait_times = [5, 10, 15, 20, 25]
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
    parser.add_argument("--benchmark", default="data/evaluation/benchmark/chemprot.json", help="输入benchmark.json文件名")
    parser.add_argument("--answer", required=True, help="待评价答案的jsonl文件名")
    parser.add_argument("--output", required=True, help="输出jsonl文件名")
    parser.add_argument("--log", default="chemprot_gptscore_test.log", help="日志文件名")
    parser.add_argument("--api_key", default=os.getenv("DASHSCOPE_API_KEY"), help="API KEY")
    parser.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="API base url")
    parser.add_argument("--threads", type=int, default=4, help="并发线程数量")
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # new prompt and examples for CPR task
    example_5 = json.dumps({
        "answer": "CPR:6",
        "score": 5,
        "reason": "Label matches gold standard. The answer is exact, concise, and fully correct, only containing the required relation type without irrelevant information.",
        "match": True
    }, ensure_ascii=False)
    example_4 = json.dumps({
        "answer": "CPR:6, which includes ANTAGONIST",
        "score": 4,
        "reason": "Label matches gold standard. The answer is correct but contains minor extra information that is not required.",
        "match": True
    }, ensure_ascii=False)
    example_3 = json.dumps({
        "answer": "CPR:6\nExplanation: This relation is antagonistic.",
        "score": 3,
        "reason": "Label matches gold standard, but the answer includes unnecessary explanation or characters.",
        "match": True
    }, ensure_ascii=False)
    example_2 = json.dumps({
        "answer": "CPR:4",
        "score": 2,
        "reason": "Label does not match gold standard, but the format is correct and the answer is related to the input sentence.",
        "match": False
    }, ensure_ascii=False)
    example_1 = json.dumps({
        "answer": "CPR:4, which includes INHIBITOR\nExplanation: The chemical inhibits the gene.",
        "score": 1,
        "reason": "Label does not match gold standard and answer contains irrelevant or excessive content.",
        "match": False
    }, ensure_ascii=False)

    system_prompt = f"""
You are an expert biomedical information extractor.
Given a task, a sentence, a model answer, and a gold standard label, rate the model's answer according to the following criteria:

Scoring reference:
- 5: Label matches gold standard, answer is concise and only contains the required relation type, no extra explanation.
- 4: Label matches gold standard, answer is correct but contains minor extra information (such as brief description or category).
- 3: Label matches gold standard, but includes unnecessary explanation or characters.
- 2: Label does not match gold standard, but the format is correct and the answer is related to the input sentence.
- 1: Label does not match gold standard and answer contains irrelevant, excessive, or meaningless content.

Output JSON fields:
- "score": integer, 1 (very poor) to 5 (excellent)
- "reason": string, explaining your score, especially the reasoning evaluation
- "match": boolean, true if the answer matches the gold label, false otherwise

Do NOT include the 'answer' field in your output JSON.  
The examples below include 'answer' for illustration only.

Output format examples:
{example_5}
{example_4}
{example_3}
{example_2}
{example_1}

Please only output the JSON object with the specified fields.
    """

    # 读取benchmark数据（适配新格式）
    with open(args.benchmark, "r", encoding="utf-8") as fin:
        benchmark_items = [json.loads(line) for line in fin if line.strip()]
    idx2benchmark = {item["idx"]: item for item in benchmark_items}

    # 读取llm答案数据
    with open(args.answer, "r", encoding="utf-8") as fin:
        answer_items = [json.loads(line) for line in fin if line.strip()]
    idx2answer = {item["idx"]: item for item in answer_items}

    # 构造所有任务
    tasks = []
    for cur_idx in sorted(idx2benchmark.keys()):
        benchmark_obj = idx2benchmark[cur_idx]
        answer_obj = idx2answer.get(cur_idx, None)
        # 提取task、input、output、sentence、gold_label
        task_desc, input_desc, output_desc, sentence, gold_label = extract_task_input_gold(benchmark_obj)
        model_answer = answer_obj.get("llm_output", "").strip() if answer_obj else ""
        tasks.append((cur_idx, task_desc, input_desc, output_desc, sentence, model_answer, gold_label))

    results = {}
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        future2idx = {
            executor.submit(
                score_one, idx, task_desc, input_desc, output_desc, sentence, model_answer, gold_label,
                system_prompt, client, args.model, 5
            ): idx
            for idx, task_desc, input_desc, output_desc, sentence, model_answer, gold_label in tasks
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

    with open(args.output, "w", encoding="utf-8") as fout:
        for cur_idx in sorted(idx2benchmark.keys()):
            out = results.get(cur_idx, {"idx": cur_idx, "error": "No result generated.", "retries": 5})
            out["idx"] = cur_idx
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
    logging.info(f"全部评分处理完成，结果保存在 {args.output}")

if __name__ == "__main__":
    main()