# from datasets import load_dataset
#
# ds = load_dataset("YufeiHFUT/bioRED")
# ds['test'].to_json('bioRED_test.json')

import json
from collections import OrderedDict

input_file = "../data/benchmark/bioRED_test.json"    # 输入文件名
output_file = "../data/benchmark/bioRED.json"  # 输出文件名

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:
    for idx, line in enumerate(fin):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        # 把idx放在最前面
        new_obj = OrderedDict()
        new_obj["idx"] = idx
        for k, v in obj.items():
            new_obj[k] = v
        fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")