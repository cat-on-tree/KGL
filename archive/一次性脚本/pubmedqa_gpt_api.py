import os
import json
from openai import OpenAI
from tqdm import tqdm
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def build_prompt(instruction, input_text, model_answer, gold_answer):
    return (
        f"Instruction:\n{instruction}\n\n"
        f"Question:\n{input_text}\n\n"
        f"Model Answer:\n{model_answer}\n\n"
        f"Gold Answer:\n{gold_answer}\n\n"
        f"Please rate the model's answer and explain your score in JSON format as instructed."
    )

def score_one(idx, instruction, input_text, model_answer, gold_answer, system_prompt, client, model, max_retries=5):
    user_prompt_full = build_prompt(instruction, input_text, model_answer, gold_answer)
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
                    "raw_output": result,
                    "gold_answer": gold_answer,
                    "model_answer": model_answer,
                    "retries": attempt+1
                }
            return {
                "idx": idx,
                "gptscore_json": result,
                "gold_answer": gold_answer,
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
        "gold_answer": gold_answer,
        "model_answer": model_answer,
        "retries": max_retries
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-max-latest", help="评价用的大模型名称")
    parser.add_argument("--answer", required=True, help="输入文件名（包含instruction/input/output/llm_output的json/jsonl）")
    parser.add_argument("--output", required=True, help="输出jsonl文件名")
    parser.add_argument("--log", default="gptscore_eval.log", help="日志文件名")
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

    # 评分标准 + 合并后的真实样例
    system_prompt = """
You are an expert biomedical information evaluator.
You will receive for each sample:
- Instruction (context/background)
- Question
- Model Answer (to be rated)
- Gold Answer (reference answer)
Please rate the model's answer according to the following criteria:
Scoring reference:
- 5: The model answer is semantically equivalent to the gold answer and provides comprehensive subgroup-specific data, additional experimental details, or mechanistic/clinical context that enriches scientific value. For example, it reports both overall and subgroup results, adds statistical methods, or discusses implications.
- 4: The model answer is semantically equivalent to the gold answer and is complete, but does not provide scientifically meaningful detail beyond the gold answer; any expansion is minor or redundant.
- 3: The model answer is close in meaning, but verbose, missing key details, or mixes relevant and irrelevant content.
- 2: The model answer is not semantically equivalent, covers little of the gold answer, or does not logically support the question.
- 1: The model answer is irrelevant, incorrect, or mainly generated content not from context, and does not logically support the gold answer.
Special notes:
- “match” is true if the scientific conclusion and reasoning are the same; false if opposite or substantially different.
- Penalize answers that introduce contradictory or incorrect science, or contradict the gold answer.
- Reward answers that provide subgroup and overall data, mechanism, or clinical context.
- Penalize verbosity and irrelevant information, unless it enhances scientific understanding.
- Answers should be data-driven, mechanistic, and domain-specific when possible.
Scoring workflow:
1. Determine semantic equivalence (“match”).
2. Assess for scientific enrichment (subgroup, mechanism, clinical impact).
3. Identify irrelevant, redundant, or contradictory content.
4. Apply scoring standard and explain reasoning.
Output a JSON object with only:
- "score": integer, 1 (very poor) to 5 (excellent)
- "reason": string, explaining your score, especially the reasoning evaluation
- "match": boolean, true if the model answer is semantically equivalent to the gold answer, false otherwise
Here are 5 examples, they share the same instruction, question and gold answer:
 Shared Context:
Programmed cell death (PCD) is the regulated death of cells within an organism. The lace plant (Aponogeton madagascariensis) produces perforations in its leaves through PCD. The leaves of the plant consist of a latticework of longitudinal and transverse veins enclosing areoles. PCD occurs in the cells at the center of these areoles and progresses outwards, stopping approximately five cells from the vasculature. The role of mitochondria during PCD has been recognized in animals; however, it has been less studied during PCD in plants. The following paper elucidates the role of mitochondrial dynamics during developmentally regulated PCD in vivo in A. madagascariensis. A single areole within a window stage leaf (PCD is occurring) was divided into three areas based on the progression of PCD; cells that will not undergo PCD (NPCD), cells in early stages of PCD (EPCD), and cells in late stages of PCD (LPCD). Window stage leaves were stained with the mitochondrial dye MitoTracker Red CMXRos and examined. Mitochondrial dynamics were delineated into four categories (M1-M4) based on characteristics including distribution, motility, and membrane potential (ΔΨm). A TUNEL assay showed fragmented nDNA in a gradient over these mitochondrial stages. Chloroplasts and transvacuolar strands were also examined using live cell imaging. The possible importance of mitochondrial permeability transition pore (PTP) formation during PCD was indirectly examined via in vivo cyclosporine A (CsA) treatment. This treatment resulted in lace plant leaves with a significantly lower number of perforations compared to controls, and that displayed mitochondrial dynamics similar to that of non-PCD cells.
Shared Question:
Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?
Shared Gold Answer:
Results depicted mitochondrial dynamics in vivo as PCD progresses within the lace plant, and highlight the correlation of this organelle with other organelles during developmental PCD. To the best of our knowledge, this is the first report of mitochondria and chloroplasts moving on transvacuolar strands to form a ring structure surrounding the nucleus during developmental PCD. Also, for the first time, we have shown the feasibility for the use of CsA in a whole plant system. Overall, our findings implicate the mitochondria as playing a critical and early role in developmentally regulated PCD in the lace plant.
Now, here are 5 model answers and their expected rating:
Example 5 (score=5):
Model Answer: Mitochondria do play a role in the remodelling of lace plant leaves during programmed cell death (PCD). The study on Aponogeton madagascariensis, a lace plant, has shown that mitochondrial dynamics are intricately involved in the PCD process. Specifically, the research indicates that the role of mitochondria in PCD is multifaceted and includes changes in mitochondrial distribution, motility, and membrane potential (ΔΨm), which are categorized into four stages (M1-M4). These dynamics are closely associated with the progression of PCD, as evidenced by the TUNEL assay, which shows fragmented nuclear DNA (nDNA) in a gradient over these mitochondrial stages. Additionally, chloroplasts and transvacuolar strands are also examined, suggesting a broader cellular context for PCD. The possible importance of mitochondrial permeability transition pore (PTP) formation during PCD is indirectly examined via in vivo cyclosporine A (CsA) treatment, leading to a significant reduction in the number of perforations and displaying mitochondrial dynamics similar to non-PCD cells. This further supports the role of mitochondrial dynamics in the regulation of PCD and the subsequent remodelling of the leaf structure.
Expected rating: {"score": 5, "reason": "The model answer is semantically equivalent to the gold answer and provides additional scientifically accurate details about mitochondrial dynamics and experimental evidence, fully covering all relevant aspects of the gold answer. The logical connection is strong and the added content enriches scientific understanding.", "match": True}
Example 4 (score=4):
Model Answer: Based on the study, mitochondria do play a role in the remodeling of lace plant leaves during programmed cell death (PCD). The research demonstrates that mitochondrial dynamics are associated with the progression of PCD in Aponogeton madagascariensis. Specifically, the mitochondria in cells undergoing PCD exhibit changes in distribution, motility, and membrane potential, which are categorized into four stages (M1-M4). These changes suggest that mitochondria are involved in the cellular processes leading to the formation of perforations in the leaves. Additionally, the TUNEL assay results show fragmented nuclear DNA (nDNA) in a gradient over these mitochondrial stages, further supporting the involvement of mitochondria in the PCD process. The study also indirectly examines the importance of mitochondrial permeability transition pore (PTP) formation during PCD, and the in vivo treatment with cyclosporine A (CsA) results in leaves with fewer perforations and mitochondrial dynamics similar to non-PCD cells, reinforcing the role of mitochondria in the remodeling of the leaves during PCD.
Expected rating: {"score": 4, "reason": "The model answer fully covers the key points of the gold answer and is scientifically accurate, but does not provide meaningful additional detail beyond the gold answer. The response is complete and relevant, with minor redundancy.", "match": True}
Example 3 (score=3):
Model Answer: Yes, mitochondria play a role in remodeling lace plant leaves during programmed cell death (PCD). The study demonstrates that mitochondrial dynamics are closely associated with the progression of PCD in *Aponogeton madagascariensis*. Key evidence includes: 1. Mitochondrial changes during PCD: Mitochondria in cells undergoing PCD (EPCD and LPCD) exhibited distinct characteristics (e.g., altered distribution, motility, and membrane potential) compared to non-PCD cells (NPCD). These changes were categorized into four stages (M1–M4), correlating with the progression of PCD. 2. Link to DNA fragmentation: TUNEL assays revealed a gradient of nuclear DNA fragmentation aligned with mitochondrial stages, suggesting mitochondrial activity influences PCD execution. 3. Role of mitochondrial permeability transition pores (PTP): Inhibition of PTP via cyclosporine A (CsA) treatment reduced perforations and altered mitochondrial dynamics to resemble those of non-PCD cells, indicating that mitochondrial permeability is critical for PCD-driven leaf remodeling. Thus, mitochondria actively participate in the PCD process, contributing to the structural changes necessary for perforation formation in lace plant leaves.
Expected rating: {"score": 3, "reason": "The model answer is close in meaning to the gold answer and covers most main points, but is verbose and mixes some relevant and irrelevant content. There is unnecessary repetition and expansion, and some details are not directly supported in the gold answer.", "match": True}
Example 2 (score=2):
Model Answer: The study mainly focuses on chloroplasts and their movement in the lace plant leaves during programmed cell death. There is little discussion about mitochondria, and their role is not clearly linked to leaf remodeling. The evidence for mitochondrial involvement is weak or absent.
Expected rating: {"score": 2, "reason": "The model answer is not semantically equivalent to the gold answer and fails to identify the main role of mitochondria in leaf remodeling during PCD. It misses essential points and provides little relevant information.", "match": False}
Example 1 (score=1):
Model Answer: Programmed cell death is a biological process found in many plants and animals. Lace plants have unique leaves. The study did not answer whether mitochondria are involved.
Expected rating: {"score": 1, "reason": "The model answer is irrelevant, does not address the question about mitochondria or leaf remodeling, and fails to cover any key points from the gold answer or context.", "match": False}
"""
    # 读取输入数据（支持jsonl或json）
    with open(args.answer, "r", encoding="utf-8") as fin:
        first = fin.read(1)
        fin.seek(0)
        if first == "[":
            items = json.load(fin)
        else:
            items = [json.loads(line) for line in fin if line.strip()]

    tasks = []
    for obj in items:
        idx = obj.get("idx")
        instruction = obj.get("instruction", "").strip()
        input_text = obj.get("input", "").strip()
        gold_answer = obj.get("output", "").strip()
        model_answer = obj.get("llm_output", "").strip()
        tasks.append((idx, instruction, input_text, model_answer, gold_answer))

    results = {}
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        future2idx = {
            executor.submit(
                score_one, idx, instruction, input_text, model_answer, gold_answer,
                system_prompt, client, args.model, 5
            ): idx
            for idx, instruction, input_text, model_answer, gold_answer in tasks
        }
        for future in tqdm(as_completed(future2idx), total=len(future2idx), desc="Concurrent LLM GPTScore rating"):
            idx = future2idx[future]
            try:
                out = future.result()
            except Exception as e:
                out = {
                    "idx": idx,
                    "error": f"Threaded error: {e}",
                    "gold_answer": "",
                    "model_answer": "",
                    "retries": 5
                }
            results[idx] = out

    # 写出结果（按输入顺序）
    with open(args.output, "w", encoding="utf-8") as fout:
        for obj in items:
            cur_idx = obj.get("idx")
            out = results.get(cur_idx, {"idx": cur_idx, "error": "No result generated.", "retries": 5})
            out["idx"] = cur_idx
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
    logging.info(f"全部评分处理完成，结果保存在 {args.output}")

if __name__ == "__main__":
    main()