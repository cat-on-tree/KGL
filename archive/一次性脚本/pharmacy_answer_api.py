import os
import json
from openai import OpenAI
from tqdm import tqdm
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def load_items(input_file):
    # 兼容扩展名为json但内容为jsonl的情况
    with open(input_file, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            # 标准json数组
            return json.load(f)
        else:
            # 每行为一个json对象（jsonl）
            return [json.loads(line) for line in f if line.strip()]

def construct_user_prompt(obj):
    question = obj.get("question", "")
    options = []
    for opt in ["A", "B", "C", "D", "E"]:
        if opt in obj and obj[opt]:
            options.append(f"{opt}. {obj[opt]}")
    options_text = "\n".join(options)
    user_prompt = f"题目：{question}\n{options_text}\n\n请给出最佳选项字母，并简要说明理由。"
    return user_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="模型名称")
    parser.add_argument("--input", default="data/evaluation/benchmark/pharmacy_test.json", help="输入文件名")
    parser.add_argument("--output", required=True, help="输出文件名")
    parser.add_argument("--log", default="med_exam_answer.log", help="日志文件名")
    parser.add_argument("--api_key", default=os.getenv("DASHSCOPE_API_KEY"), help="API KEY")
    parser.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="API base url")
    parser.add_argument("--threads", type=int, default=4, help="并发线程数，默认单线程")
    parser.add_argument("--max_retries", type=int, default=5, help="每条最大重试次数")
    parser.add_argument("--retry_base_wait", type=float, default=20, help="429出错后等待的基础秒数，指数退避")
    parser.add_argument("--enable_thinking", action="store_true", help="启用thinking模式（即请求时传extra_body={enable_thinking:False}）")
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    items = load_items(args.input)

    def process_item(idx, obj):
        max_retries = args.max_retries
        retry_base_wait = args.retry_base_wait

        if "idx" not in obj:
            msg = "原始数据缺少idx字段，跳过。"
            logging.warning(msg)
            return {"error": msg}

        out_idx = obj["idx"]
        system_prompt = "你是执业药师考试知识问答 AI，请根据题目和选项，给出最佳答案字母，并简要说明理由。"
        user_prompt = construct_user_prompt(obj)

        for attempt in range(max_retries):
            try:
                kwargs = dict(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                )
                if args.enable_thinking:
                    kwargs["extra_body"] = {"enable_thinking": False}
                completion = client.chat.completions.create(**kwargs)

                result = completion.choices[0].message.content
                logging.info(f"第{out_idx + 1}条成功生成。")
                return {
                    "idx": out_idx,
                    "system": system_prompt,
                    "user": user_prompt,
                    "llm_output": result,
                    "question": obj.get("question", ""),
                    "answer": obj.get("answer", ""),
                    "options": {opt: obj.get(opt, "") for opt in ["A", "B", "C", "D", "E"]},
                    "source": obj.get("source", ""),
                    "question_type": obj.get("question_type", "")
                }
            except Exception as e:
                wait_time = retry_base_wait * (2 ** attempt)
                errmsg = f"第{out_idx + 1}条请求出错：{e}，第{attempt + 1}次重试，等待{wait_time:.1f}秒..."
                print(errmsg)
                logging.warning(errmsg)
                time.sleep(wait_time)
                continue
        errmsg = f"第{out_idx + 1}条请求连续{max_retries}次失败，已跳过。最后错误：{e}"
        print(errmsg)
        logging.error(errmsg)
        return {"idx": out_idx, "error": errmsg}

    if args.threads == 1:
        with open(args.output, "w", encoding="utf-8") as fout:
            for idx, obj in enumerate(tqdm(items, desc="LLM生成中")):
                out = process_item(idx, obj)
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
        logging.info(f"全部处理完成，结果保存在 {args.output}")

    else:
        results = {}
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(process_item, idx, obj) for idx, obj in enumerate(items)]
            for future in tqdm(as_completed(futures), total=len(items), desc="LLM生成中(并发)"):
                out = future.result()
                if out is not None and "idx" in out:
                    results[out["idx"]] = out

        with open(args.output, "w", encoding="utf-8") as fout:
            for idx in sorted(results):
                out = results[idx]
                if out is None:
                    continue
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
        logging.info(f"全部处理完成(并发)，结果保存在 {args.output}")

if __name__ == "__main__":
    main()