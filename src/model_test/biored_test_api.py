import os
import json
from openai import OpenAI
from tqdm import tqdm
import argparse
import logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-max-latest", help="模型名称")
    parser.add_argument("--input", required=True, help="输入文件名")
    parser.add_argument("--output", required=True, help="输出文件名")
    parser.add_argument("--log", default="biored-test.log", help="日志文件名")
    parser.add_argument("--api_key", default=os.getenv("DASHSCOPE_API_KEY"), help="API KEY")
    parser.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="API base url")
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # 读取jsonl文件
    with open(args.input, "r", encoding="utf-8") as fin:
        items = [json.loads(line) for line in fin if line.strip()]

    idx2out = {}
    for idx, obj in enumerate(tqdm(items, desc="Refine label by LLM")):
        cur_idx = obj.get("idx", idx)

        # 提取system字段至"based only on the information provided in the text"为止
        system_full = obj.get("system", "").strip()
        end_flag = "based only on the information provided in the text."
        pos = system_full.find(end_flag)
        if pos != -1:
            system_part = system_full[:pos + len(end_flag)]
        else:
            system_part = system_full

        # 拼接新system_prompt
        system_prompt = (
            system_part +
            '\nonly output a json contains a label, the output format examples are as follows: { "label": "Association" }; { "label": "bind" }; { "label": "Cotreatment" }'
        )

        user_prompt = obj.get("llm_output", "").strip()
        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            result = completion.choices[0].message.content
            logging.info(f"第{cur_idx + 1}条成功生成。")
            out = {
                "idx": cur_idx,
                "label_json": result
            }
        except Exception as e:
            errmsg = f"第{cur_idx + 1}条请求出错：{e}"
            print(errmsg)
            logging.error(errmsg)
            out = {
                "idx": cur_idx,
                "error": errmsg
            }
        idx2out[cur_idx] = out

    with open(args.output, "w", encoding="utf-8") as fout:
        for obj in items:
            cur_idx = obj.get("idx")
            out = idx2out.get(cur_idx, {"idx": cur_idx, "error": "No result generated."})
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
    logging.info(f"全部处理完成，结果保存在 {args.output}")

if __name__ == "__main__":
    main()