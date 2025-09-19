####数据集下载
# from datasets import load_dataset
#
# ds = load_dataset("YufeiHFUT/bioRED")
# ds['test'].to_json('bioRED_test.json')

# from datasets import load_dataset
#
# ds = load_dataset("clinicalnlplab/chemprot_test")
# ds['train'].to_json('chemprot_test.json')
# from datasets import load_dataset
#
# ds = load_dataset("kroshan/BioASQ")
# ds['train'].to_json('BioASQ_test.json')

# import os
# print(os.environ.get("HF_ENDPOINT"))
#
# from datasets import load_dataset
# ds = load_dataset("hiyouga/math12k")

import pandas as pd
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('hiyouga/PubMedQA', subset_name='default', split='test')
df = pd.DataFrame(ds)
df['idx'] = range(len(df))
cols = ['idx'] + [c for c in df.columns if c != 'idx']
df = df[cols]
df.to_json('pubmedqa.json', orient='records', lines=True, force_ascii=False)