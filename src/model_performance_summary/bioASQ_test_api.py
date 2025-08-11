import os
import json
import re
from openai import OpenAI
from tqdm import tqdm
import argparse
import logging

def extract_gold_answer(text):
    """
    从text字段中提取<answer> ... 部分，去除前后空格和标点
    """
    m = re.search(r"<answer>\s*([^\n<]+)", text)
    if m:
        # 去除末尾可能的标点和空格
        answer = m.group(1).strip().strip(".;，。；")
        return answer
    return ""

def extract_predicted_answer(llm_output):
    """
    从llm_output中提取'Answer to Question 1:'和'Answer to Question 2'之间的内容
    """
    # 支持不同可能的分隔符
    m = re.search(r"Answer to Question 1:(.*?)(?:Answer to Question 2|\Z)", llm_output, re.DOTALL | re.IGNORECASE)
    if m:
        answer = m.group(1).strip()
        # 去除多余的前后标点和空格
        answer = answer.strip(":：.。；;，, \n\t")
        return answer
    return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-max-latest", help="模型名称")
    parser.add_argument("--input", required=True, help="输入文件名（包含 LLM 预测答案的jsonl）")
    parser.add_argument("--gold", default="data/evaluation/benchmark/BioASQ.json", help="gold标准文件名（含question和gold answer的json或jsonl）")
    parser.add_argument("--output", required=True, help="输出文件名")
    parser.add_argument("--log", default="bioASQ-test.log", help="日志文件名")
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

    # 读取gold标准文件，支持json或jsonl
    gold_path = args.gold
    with open(gold_path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            gold_items = json.load(f)
        else:
            gold_items = [json.loads(line) for line in f if line.strip()]
    idx2gold = {obj["idx"]: obj for obj in gold_items if "idx" in obj}

    # 读取llm预测输出
    with open(args.input, "r", encoding="utf-8") as fin:
        items = [json.loads(line) for line in fin if line.strip()]

    idx2out = {}
    for idx, obj in enumerate(tqdm(items, desc="LLM判断正误")):
        cur_idx = obj.get("idx", idx)
        gold_obj = idx2gold.get(cur_idx)
        if not gold_obj:
            logging.warning(f"没有找到gold标准，idx={cur_idx}")
            out = {
                "idx": cur_idx,
                "error": "No gold standard found for idx."
            }
            idx2out[cur_idx] = out
            continue

        question = gold_obj.get("question", "").strip()
        gold_answer = extract_gold_answer(gold_obj.get("text", ""))
        llm_output = obj.get("llm_output", "")

        predicted_answer = extract_predicted_answer(llm_output)
        if not predicted_answer:
            logging.warning(f"无法提取预测答案, idx={cur_idx}")
            out = {
                "idx": cur_idx,
                "error": "No predicted answer found."
            }
            idx2out[cur_idx] = out
            continue

        # 构造system和user prompt
        system_prompt = (
            "You are an expert in biomedical text analysis, specifically in the BioASQ challenge. "
            "Your task is: given a question and two answers, which contains a gold answer and a predicted answer. "
            "Determine if the predicted answer is the same to the label.\n"
            "If yes, output true, otherwise output false.\n"
            "Here is an example:\n"
            "Question: What is the gene mutated in the Gaucher disease?\n"
            "Gold Answer: glucocerebrosidase.\n"
            "Predicted answer: The gene mutated in Gaucher's disease is glucocerebrosidase (GBA1).\n"
            "Your answer is: { \"label\": \"True\" }\n"
            "only output a json contains a label, the output format examples are as follows:\n"
            "{ \"label\": \"True\" }; { \"label\": \"False\" }\n"
        )
        user_prompt = (
            f"Question: {question}\n"
            f"Gold Answer: {gold_answer}\n"
            f"Predicted answer: {predicted_answer}\n"
        )

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
                "label_json": result,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer
            }
        except Exception as e:
            errmsg = f"第{cur_idx + 1}条请求出错：{e}"
            print(errmsg)
            logging.error(errmsg)
            out = {
                "idx": cur_idx,
                "error": errmsg,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer
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