import os
import json
import re
from openai import OpenAI
from tqdm import tqdm
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def load_items(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def get_done_set(output_path):
    done_set = set()
    if not os.path.exists(output_path):
        return done_set
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if "idx" in obj:
                    done_set.add(obj["idx"])
            except Exception:
                continue
    return done_set

def str2bool_or_none(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ('yes', 'true', 't', 'y', '1'):
        return True
    if v in ('no', 'false', 'f', 'n', '0'):
        return False
    return None

def chat_completion_with_stream(client, **kwargs):
    stream_options = kwargs.pop("stream_options", None)
    completion = client.chat.completions.create(stream=True, **kwargs)
    full_content = ""
    for chunk in completion:
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                full_content += delta.content
    return full_content

def chat_completion_with_sync(client, **kwargs):
    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content.strip()

def process_item(idx, obj, args, client):
    max_retries = args.max_retries
    retry_base_wait = args.retry_base_wait

    if "idx" not in obj:
        msg = "原始数据缺少idx字段，跳过。"
        logging.warning(msg)
        return {"error": msg}

    out_idx = obj["idx"]
    instruction = obj.get("instruction", "")
    user_input = obj.get("input", "")

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input}
    ]

    for attempt in range(max_retries):
        try:
            kwargs = dict(model=args.model, messages=messages)
            if args.enable_thinking is not None:
                kwargs["extra_body"] = {"enable_thinking": args.enable_thinking}
                if args.enable_thinking:
                    answer = chat_completion_with_stream(client, **kwargs)
                else:
                    answer = chat_completion_with_sync(client, **kwargs)
            else:
                answer = chat_completion_with_sync(client, **kwargs)
            answer = answer.strip()
            break
        except Exception as e:
            wait_time = retry_base_wait * (2 ** attempt)
            errmsg = f"第{out_idx}条请求出错：{e}，第{attempt + 1}次重试，等待{wait_time:.1f}秒..."
            print(errmsg)
            logging.warning(errmsg)
            time.sleep(wait_time)
    else:
        errmsg = f"第{out_idx}条请求连续{max_retries}次失败，已跳过。最后错误：{e}"
        print(errmsg)
        logging.error(errmsg)
        return {"idx": out_idx, "error": errmsg}

    logging.info(f"第{out_idx}条成功生成。")
    out = obj.copy()
    out["llm_output"] = answer
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="模型名称")
    parser.add_argument("--input", default="data/evaluation/benchmark/pubmedqa.json", help="输入文件名")
    parser.add_argument("--output", required=True, help="输出文件名")
    parser.add_argument("--log", default="api-infer.log", help="日志文件名")
    parser.add_argument("--api_key", default=os.getenv("DASHSCOPE_API_KEY"), help="API KEY")
    parser.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="API base url")
    parser.add_argument("--threads", type=int, default=4, help="并发线程数，默认单线程")
    parser.add_argument("--max_retries", type=int, default=5, help="每条最大重试次数")
    parser.add_argument("--retry_base_wait", type=float, default=20, help="429出错后等待的基础秒数，指数退避")
    parser.add_argument("--enable_thinking", nargs='?', const=False, default=None, help="enable_thinking，True/False，不传则None，只写参数名为False")
    args = parser.parse_args()

    args.enable_thinking = str2bool_or_none(args.enable_thinking)

    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    items = load_items(args.input)

    done_set = get_done_set(args.output)
    print(f"检测到已完成 {len(done_set)} 条，将跳过这些样本...")

    if args.threads == 1:
        with open(args.output, "a", encoding="utf-8") as fout:
            for idx, obj in enumerate(tqdm(items, desc="LLM生成中")):
                if obj.get("idx") in done_set:
                    continue
                out = process_item(idx, obj, args, client)
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
        logging.info(f"全部处理完成，结果保存在 {args.output}")

    else:
        to_process = [(idx, obj) for idx, obj in enumerate(items) if obj.get("idx") not in done_set]
        results = {}
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(process_item, idx, obj, args, client) for idx, obj in to_process]
            for future in tqdm(as_completed(futures), total=len(to_process), desc="LLM生成中(并发)"):
                out = future.result()
                if out is not None and "idx" in out:
                    results[out["idx"]] = out

        with open(args.output, "a", encoding="utf-8") as fout:
            for idx in sorted(results):
                out = results[idx]
                if out is None:
                    continue
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
        logging.info(f"全部处理完成(并发)，结果保存在 {args.output}")

if __name__ == "__main__":
    main()