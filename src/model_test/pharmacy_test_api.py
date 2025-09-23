import os
import json
from openai import OpenAI
from tqdm import tqdm
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def construct_analysis_prompt(obj):
    """
    构造用于让LLM判断选择了哪个选项的prompt，包括instruction、选项和llm_output
    """
    question = obj.get("question", "")
    options = []
    for opt in ["A", "B", "C", "D", "E"]:
        if opt in obj and obj[opt]:
            options.append(f"{opt}. {obj[opt]}")
    options_text = "\n".join(options)
    llm_output = obj.get("llm_output", "").strip()

    analysis_instruction = (
        f"你是一个信息提取者，接下来你会看到一个题目及一些选项，以及一份他人做出的回答：\n\n"
        f"题目：{question}\n选项：{options_text}\n\n"
        f"他人回答如下：\n{llm_output}\n\n"
        "请你只判断模型最终选择了哪个选项（A/B/C/D/E），不要输出任何解释。"
        '输出json格式，格式为：{"label": "X"}，其中X为模型选择的答案。'
    )
    return analysis_instruction

def process_one(idx, obj, client, args, max_retries=5):
    cur_idx = obj.get("idx", idx)
    analysis_prompt = construct_analysis_prompt(obj)
    answer = obj.get("answer", "")

    wait_times = [5, 10, 15, 20, 25]
    last_exception = None
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": analysis_prompt},
                    {"role": "user", "content": ""},
                ],
                response_format={"type": "json_object"},
            )
            result = completion.choices[0].message.content
            # 解析 LLM 输出的 json 字符串
            label = None
            try:
                label_obj = json.loads(result)
                label = label_obj.get("label", None)
            except Exception as e:
                label = None
                logging.warning(f"第{cur_idx + 1}条LLM输出无法解析为json: {result}, 错误: {e}")
            out = {
                "idx": cur_idx,
                "label": label,
                "answer": answer,
                "retries": attempt + 1
            }
            logging.info(f"第{cur_idx + 1}条成功生成。")
            return cur_idx, out
        except Exception as e:
            last_exception = e
            errmsg = f"第{cur_idx + 1}条第{attempt+1}次请求出错：{e}"
            print(errmsg)
            logging.error(errmsg)
            if attempt < max_retries - 1:
                time.sleep(wait_times[attempt])
    out = {
        "idx": cur_idx,
        "label": None,
        "answer": answer,
        "error": f"请求重试{max_retries}次仍失败: {last_exception}",
        "retries": max_retries
    }
    return cur_idx, out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-max-latest", help="模型名称")
    parser.add_argument("--input", required=True, help="输入文件名")
    parser.add_argument("--output", required=True, help="输出文件名")
    parser.add_argument("--log", default="extract-choice-label.log", help="日志文件名")
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

    with open(args.input, "r", encoding="utf-8") as fin:
        items = [json.loads(line) for line in fin if line.strip()]

    idx2out = {}
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [
            executor.submit(process_one, idx, obj, client, args)
            for idx, obj in enumerate(items)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extract choice label by LLM (parallel)"):
            cur_idx, out = future.result()
            idx2out[cur_idx] = out

    with open(args.output, "w", encoding="utf-8") as fout:
        for obj in items:
            cur_idx = obj.get("idx")
            out = idx2out.get(cur_idx, {"idx": cur_idx, "label": None, "answer": obj.get("answer", None), "error": "No result generated."})
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
    logging.info(f"全部处理完成，结果保存在 {args.output}")

if __name__ == "__main__":
    main()