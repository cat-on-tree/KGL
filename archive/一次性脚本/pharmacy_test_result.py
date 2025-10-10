import json
from sklearn.metrics import classification_report
import argparse

def main(json_file, report_file):
    preds, gts = [], []
    with open(json_file, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            pred_label = obj.get("label")
            gt_label = obj.get("answer")
            preds.append(pred_label if pred_label is not None else "None")
            gts.append(gt_label if gt_label is not None else "None")

    # 统计所有出现过的label
    labels = sorted(list(set(gts) | set(preds)))

    report = classification_report(gts, preds, labels=labels, digits=2, zero_division=0)

    print("==== Classification Report ====")
    print(report)

    with open(report_file, "w", encoding="utf-8") as fout:
        fout.write("==== Classification Report ====\n")
        fout.write(report)
    print(f"\n分类报告已保存到 {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="包含label和answer的jsonl文件")
    parser.add_argument("--result", default="classification_report.txt", help="评测报告输出txt文件")
    args = parser.parse_args()
    main(args.test, args.result)