import os
import json
import re
from tqdm import tqdm
import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_items(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
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

def extract_assistant_response(result):
    match = re.search(r"(?:^|\n)assistant\s*\n(.*)", result, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>", result, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return result.strip()

def remove_think_tags(text):
    return re.sub(r"<think>\s*?</think>\s*", "", text, flags=re.DOTALL)

def local_generate(model, tokenizer, system_prompt, user_prompt, max_new_tokens=4096, device="cpu", enable_thinking=None, use_qwen_template=False, stream=False):
    # Qwen/chat模板兼容
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
    assistant_result = remove_think_tags(assistant_result)
    return assistant_result

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="本地transformers模型目录")
    parser.add_argument("--input", default="data/evaluation/benchmark/pubmedqa.json", help="输入文件名")
    parser.add_argument("--output", required=True, help="输出文件名")
    parser.add_argument("--log", default="infer.log", help="日志文件名")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--device", default="cpu", help="cpu、cuda 或 mps")
    parser.add_argument("--enable_thinking", nargs='?', const=False, default=None, help="enable_thinking，True/False，不传则None，只写参数名为False")
    parser.add_argument("--stream", action="store_true", help="强制流式解码（如模型支持），否则仅在enable_thinking为True时自动流式")
    args = parser.parse_args()

    args.enable_thinking = str2bool_or_none(args.enable_thinking)

    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    items = load_items(args.input)
    done_set = get_done_set(args.output)
    print(f"检测到已完成 {len(done_set)} 条，将跳过这些样本...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    if device == "mps":
        model = model.to(torch.float32)
    model = model.to(device).eval()

    model_type = getattr(model.config, "model_type", "").lower()
    use_qwen_template = "qwen" in model_type

    with open(args.output, "a", encoding="utf-8") as fout:
        for idx, obj in enumerate(tqdm(items, desc="本地模型单轮推理中")):
            if "idx" not in obj:
                msg = "原始数据缺少idx字段，跳过。"
                logging.warning(msg)
                continue
            out_idx = obj["idx"]
            if out_idx in done_set:
                continue

            instruction = obj.get("instruction", "")
            user_input = obj.get("input", "")
            system_prompt = instruction
            user_prompt = user_input

            try:
                answer = local_generate(
                    model, tokenizer,
                    system_prompt, user_prompt,
                    max_new_tokens=args.max_new_tokens, device=device,
                    enable_thinking=args.enable_thinking,
                    use_qwen_template=use_qwen_template,
                    stream=args.stream or (args.enable_thinking is True)
                )
                answer = answer.strip()
                logging.info(f"第{out_idx}条本地模型生成成功。")
            except Exception as e:
                errmsg = f"第{out_idx}条本地模型推理出错：{e}"
                print(errmsg)
                logging.error(errmsg)
                continue

            out = obj.copy()
            out["llm_output"] = answer
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

    logging.info(f"全部处理完成，结果保存在 {args.output}")

if __name__ == "__main__":
    main()