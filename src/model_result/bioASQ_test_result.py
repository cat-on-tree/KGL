import json
from sklearn.metrics import classification_report, accuracy_score
import argparse

def extract_label(label_json_str):
    """从label_json中提取label的True/False字符串"""
    try:
        if isinstance(label_json_str, dict):
            return label_json_str.get("label")
        elif isinstance(label_json_str, str):
            return json.loads(label_json_str).get("label")
    except Exception:
        return None

def main(input_file, report_file):
    labels = []
    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            label = extract_label(obj.get("label_json"))
            if label is not None:
                labels.append(label)

    # 统计True/False数量和比例
    n_true = labels.count("True")
    n_false = labels.count("False")
    total = len(labels)
    accuracy = n_true / total if total > 0 else 0

    print("Label统计：")
    print(f"True: {n_true}")
    print(f"False: {n_false}")
    print(f"总数: {total}")
    print(f"Accuracy (True占比): {accuracy:.4f}")

    # 由于没有gold标准，无法算precision/recall，只能统计分布
    with open(report_file, "w", encoding="utf-8") as fout:
        fout.write("Label统计：\n")
        fout.write(f"True: {n_true}\n")
        fout.write(f"False: {n_false}\n")
        fout.write(f"总数: {total}\n")
        fout.write(f"Accuracy (True占比): {accuracy:.4f}\n")
    print(f"\n统计结果已保存到 {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="模型输出jsonl文件（含label_json字段）")
    parser.add_argument("--result", default="label_report.txt", help="统计报告输出txt文件")
    args = parser.parse_args()
    main(args.test, args.result)