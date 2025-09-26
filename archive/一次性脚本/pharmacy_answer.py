import os
import json
from tqdm import tqdm
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def resolve_device(device_str):
    device_str = device_str.lower()
    if device_str == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif device_str == "mps" and torch.backends.mps.is_available():
        return "mps"
    elif device_str == "cpu":
        return "cpu"
    else:
        print(f"Warning: Requested device '{device_str}' not available. Falling back to CPU.")
        return "cpu"

def extract_assistant_response(result):
    """从模型输出中提取assistant回复，兼容多种格式"""
    import re
    match = re.search(r"(?:^|\n)assistant\s*\n(.*)", result, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>", result, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return result.strip()

def local_generate(model, tokenizer, system_prompt, user_prompt, max_new_tokens=4096, device="cpu", use_qwen_template=False, enable_thinking=None):
    # Qwen系列支持chat_template，否则直接拼接
    if use_qwen_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        chat_template_kwargs = dict(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # 只在Qwen chat模板支持的情况下传递enable_thinking参数
        if enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = enable_thinking
        prompt = tokenizer.apply_chat_template(**chat_template_kwargs)
    else:
        prompt = system_prompt + "\n" + user_prompt

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_result = extract_assistant_response(result)
    return assistant_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="本地transformers模型目录")
    parser.add_argument("--input", default="data/evaluation/benchmark/pharmacy_test.json", help="输入文件名")
    parser.add_argument("--output", required=True, help="输出文件名")
    parser.add_argument("--log", default="med_exam_answer.log", help="日志文件名")
    parser.add_argument("--threads", type=int, default=4, help="并发线程数，默认4")
    parser.add_argument("--max_retries", type=int, default=5, help="每条最大重试次数")
    parser.add_argument("--retry_base_wait", type=float, default=10, help="出错后等待的基础秒数，指数退避")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--device", default="cpu", help="cpu、cuda 或 mps")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="出现此参数时，显式关闭思考模式（即传 enable_thinking=False），不传则保持模型默认")

    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    items = load_items(args.input)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    if device == "mps":
        model = model.to(torch.float32)
    model = model.to(device).eval()

    # 判断是否Qwen系列
    model_type = getattr(model.config, "model_type", "").lower()
    use_qwen_template = "qwen" in model_type

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
        last_error = None

        # 判断是否需要传enable_thinking参数
        enable_thinking = None
        if args.enable_thinking:
            enable_thinking = False

        for attempt in range(max_retries):
            try:
                result = local_generate(
                    model, tokenizer, system_prompt, user_prompt,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                    use_qwen_template=use_qwen_template,
                    enable_thinking=enable_thinking
                )
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
                last_error = e
                wait_time = retry_base_wait * (2 ** attempt)
                errmsg = f"第{out_idx + 1}条本地模型推理出错：{e}，第{attempt + 1}次重试，等待{wait_time:.1f}秒..."
                print(errmsg)
                logging.warning(errmsg)
                time.sleep(wait_time)
                continue
        errmsg = f"第{out_idx + 1}条本地模型推理连续{max_retries}次失败，已跳过。最后错误：{last_error}"
        print(errmsg)
        logging.error(errmsg)
        return {"idx": out_idx, "error": errmsg}

    # 支持多线程推理
    if args.threads == 1:
        with open(args.output, "w", encoding="utf-8") as fout:
            for idx, obj in enumerate(tqdm(items, desc="本地模型推理中")):
                out = process_item(idx, obj)
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
        logging.info(f"全部处理完成，结果保存在 {args.output}")
    else:
        results = {}
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(process_item, idx, obj) for idx, obj in enumerate(items)]
            for future in tqdm(as_completed(futures), total=len(items), desc="本地模型推理中(并发)"):
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