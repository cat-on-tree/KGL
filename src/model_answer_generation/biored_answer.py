import os
import json
import re
from tqdm import tqdm
import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def extract_prompts(text):
    sys_match = re.search(r"<s>\[INST\]<<SYS>>\n(.*?)\n<<\/SYS>>\n", text, re.DOTALL)
    user_match = re.search(r"<<\/SYS>>\n\n(.*)\[\/INST\]", text, re.DOTALL)
    if sys_match and user_match:
        system_prompt = sys_match.group(1).strip()
        user_prompt = user_match.group(1).strip()
        return system_prompt, user_prompt
    return None, None

def load_items(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]

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

def local_generate(model, tokenizer, system_prompt, user_prompt, max_new_tokens=128, device="cpu"):
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
    # 去掉prompt本身，只保留模型生成的内容（可根据实际需要调整）
    if result.startswith(prompt):
        result = result[len(prompt):].strip()
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="本地transformers模型目录")
    parser.add_argument("--input", default="data/evaluation/benchmark/bioRED.json", help="输入文件名")
    parser.add_argument("--output", required=True, help="输出文件名")
    parser.add_argument("--log", default="biored-answer.log", help="日志文件名")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", default="cpu", help="cpu、cuda 或 mps")
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
    # mps 设备只支持float32或float16，可以适当转换
    if device == "mps":
        model = model.to(torch.float32)
    model = model.to(device).eval()

    with open(args.output, "w", encoding="utf-8") as fout:
        for idx, obj in enumerate(tqdm(items, desc="本地模型推理中")):
            if "idx" not in obj:
                msg = "原始数据缺少idx字段，跳过。"
                logging.warning(msg)
                continue
            out_idx = obj["idx"]
            text = obj["data"]
            sys_prompt, user_prompt = extract_prompts(text)
            if not (sys_prompt and user_prompt):
                msg = f"第{out_idx + 1}条未能正确抽取prompt，跳过。"
                logging.warning(msg)
                continue
            try:
                result = local_generate(model, tokenizer, sys_prompt, user_prompt, max_new_tokens=args.max_new_tokens, device=device)
                logging.info(f"第{out_idx + 1}条本地模型生成成功。")
                out = {
                    "idx": out_idx,
                    "system": sys_prompt,
                    "user": user_prompt,
                    "llm_output": result
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
            except Exception as e:
                errmsg = f"第{out_idx + 1}条本地模型推理出错：{e}"
                print(errmsg)
                logging.error(errmsg)
                continue
    logging.info(f"全部处理完成，结果保存在 {args.output}")

if __name__ == "__main__":
    main()