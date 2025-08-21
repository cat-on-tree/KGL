import os
import json
from openai import OpenAI
from tqdm import tqdm
import argparse
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def extract_answer_context_gold(item):
    """
    从benchmark每条item中提取问题、参考文本、金标准答案、金标准支持句
    """
    question = item.get("question", "")
    text = item.get("text", "")
    answer_match = re.search(r"<answer>\s*([^\n<]+)", text)
    context_match = re.search(r"<context>\s*([^\n]+)", text)
    gold_answer = answer_match.group(1).strip() if answer_match else ""
    context = context_match.group(1).strip() if context_match else ""
    supporting_sentences = item.get("supporting_sentences", [])
    gold_support = supporting_sentences[0].strip() if supporting_sentences else ""
    return question, context, gold_answer, gold_support

def extract_llm_answer_and_support(llm_output):
    """
    从llm输出中提取模型回答和支持句
    """
    answer_match = re.search(r"Answer to Question 1:(.*?)(?:Answer to Question 2|\Z)", llm_output, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""
    support_match = re.search(r"Answer to Question 2.*?:\s*(.*)", llm_output, re.DOTALL)
    supporting_sentence = support_match.group(1).strip() if support_match else ""
    return answer, supporting_sentence

def build_prompt(question, context, gold_answer, gold_support, model_answer, model_support):
    return (
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        f"Model Answer:\n{model_answer}\n\n"
        f"Model Supporting Sentence:\n{model_support}\n\n"
        f"Gold Answer:\n{gold_answer}\n\n"
        f"Gold Supporting Sentence:\n{gold_support}\n\n"
        f"Please rate the model's answer and supporting sentence following the instructions below and explain your score in JSON format."
    )

def score_one(idx, question, context, gold_answer, gold_support, model_answer, model_support, system_prompt, client, model, max_retries=5):
    user_prompt_full = build_prompt(question, context, gold_answer, gold_support, model_answer, model_support)
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
                    "gold_answer": gold_answer,
                    "gold_support": gold_support,
                    "model_answer": model_answer,
                    "model_support": model_support,
                    "retries": attempt+1
                }
            return {
                "idx": idx,
                "gptscore_json": result,
                "gold_answer": gold_answer,
                "gold_support": gold_support,
                "model_answer": model_answer,
                "model_support": model_support,
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
        "gold_answer": gold_answer,
        "gold_support": gold_support,
        "model_answer": model_answer,
        "model_support": model_support,
        "retries": max_retries
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-max-latest", help="评价用的大模型名称")
    parser.add_argument("--benchmark", default="data/evaluation/benchmark/BioASQ.json", help="输入benchmark.json文件名")
    parser.add_argument("--answer", required=True, help="待评价答案的jsonl文件名")
    parser.add_argument("--output", required=True, help="输出jsonl文件名")
    parser.add_argument("--log", default="bioASQ_gptscore_test.log", help="日志文件名")
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

    # 放宽标准的few-shot评分示例
    example_5 = json.dumps({
        "answer": "glucocerebrosidase",
        "supporting_sentence": "Gaucher's disease (GD) results from a deficiency of the lysosomal enzyme glucocerebrosidase.",
        "score": 5,
        "reason": "Answer matches gold standard exactly; supporting sentence is concise, directly relevant and covers the answer logically.",
        "match": True
    }, ensure_ascii=False)
    example_4 = json.dumps({
        "answer": "glucocerebrosidase (GBA1)",
        "supporting_sentence": "Gaucher's disease results from a deficiency of glucocerebrosidase. The complexity of identification and characterization of mutations in the gene of glucocerebrosidase (GBA1) is caused by a great amount of mutated alleles.",
        "score": 4,
        "reason": "Answer matches gold standard; supporting sentence contains relevant information, with slight redundancy but overall logical support.",
        "match": True
    }, ensure_ascii=False)
    example_3 = json.dumps({
        "answer": "The gene mutated in Gaucher disease is glucocerebrosidase.",
        "supporting_sentence": "Gaucher's disease is caused by a deficiency of glucocerebrosidase. The context mentions both glucocerebrosidase and saposin C.",
        "score": 3,
        "reason": "Answer is correct but verbose; supporting sentence is somewhat lengthy or mixes relevant and irrelevant facts. Logic is reasonable but could be more concise.",
        "match": True
    }, ensure_ascii=False)
    example_2 = json.dumps({
        "answer": "saposin C",
        "supporting_sentence": "Gaucher's disease may also result from a deficiency of saposin C.",
        "score": 2,
        "reason": "Answer does not match gold standard but is present in context; supporting sentence is relevant but does not cover gold answer.",
        "match": False
    }, ensure_ascii=False)
    example_1 = json.dumps({
        "answer": "Gaucher disease is a lysosomal storage disorder.",
        "supporting_sentence": "It is a lysosomal disorder. Mutations may be complex.",
        "score": 1,
        "reason": "Answer and supporting sentence are irrelevant or overly verbose and do not logically cover the gold answer.",
        "match": False
    }, ensure_ascii=False)

    system_prompt = f"""
You are an expert biomedical information extractor.
Given a question, a context (background knowledge), a model answer, a supporting sentence, a gold answer, and a gold supporting sentence, rate the model's answer and supporting sentence according to the following criteria:

Scoring reference:
- 5: Answer matches gold standard exactly; supporting sentence is concise, directly relevant, covers the answer with clear and logical reasoning. Minor extra details are acceptable if logic is strong and no excessive content.
- 4: Answer matches gold standard; supporting sentence is relevant and covers the answer, but contains some redundancy or slight verbosity.
- 3: Answer matches gold standard or is very close; supporting sentence is somewhat lengthy or mixes relevant and irrelevant content, but covers the answer. Logic is reasonable but could be more concise.
- 2: Answer does not match gold standard but is present in context; supporting sentence is relevant but does not cover gold answer, or is too verbose.
- 1: Answer and supporting sentence are irrelevant, incorrect, or overly verbose (e.g., quoting two、three or more sentences when one suffices), and do not logically support the gold answer.

Output JSON fields:
- "score": integer, 1 (very poor) to 5 (excellent)
- "reason": string, explaining your score, especially the reasoning evaluation
- "match": boolean, true if answer matches gold standard, false otherwise

Do NOT include the 'answer' or 'supporting_sentence' fields in your output JSON.  
The examples below include them for illustration only.

Output format examples:
{example_5}
{example_4}
{example_3}
{example_2}
{example_1}

Please only output the JSON object with the specified fields.
    """

    # 读取benchmark数据（新结构）
    with open(args.benchmark, "r", encoding="utf-8") as fin:
        benchmark_items = [json.loads(line) for line in fin if line.strip()]
    idx2benchmark = {item["idx"]: item for item in benchmark_items}

    # 读取llm答案数据（新结构）
    with open(args.answer, "r", encoding="utf-8") as fin:
        answer_items = [json.loads(line) for line in fin if line.strip()]
    idx2answer = {item["idx"]: item for item in answer_items}

    # 构造所有任务
    tasks = []
    for cur_idx in sorted(idx2benchmark.keys()):
        benchmark_obj = idx2benchmark[cur_idx]
        answer_obj = idx2answer.get(cur_idx, None)
        question, context, gold_answer, gold_support = extract_answer_context_gold(benchmark_obj)
        llm_raw_output = answer_obj.get("llm_output", "") if answer_obj else ""
        model_answer, model_support = extract_llm_answer_and_support(llm_raw_output)
        tasks.append((cur_idx, question, context, gold_answer, gold_support, model_answer, model_support))

    results = {}
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        future2idx = {
            executor.submit(
                score_one, idx, question, context, gold_answer, gold_support, model_answer, model_support,
                system_prompt, client, args.model, 5
            ): idx
            for idx, question, context, gold_answer, gold_support, model_answer, model_support in tasks
        }
        for future in tqdm(as_completed(future2idx), total=len(future2idx), desc="Concurrent LLM scoring w/ retry"):
            idx = future2idx[future]
            try:
                out = future.result()
            except Exception as e:
                out = {
                    "idx": idx,
                    "error": f"Threaded error: {e}",
                    "gold_answer": "",
                    "gold_support": "",
                    "model_answer": "",
                    "model_support": "",
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