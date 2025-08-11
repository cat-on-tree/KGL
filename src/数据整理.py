####数据集下载
# from datasets import load_dataset
#
# ds = load_dataset("YufeiHFUT/bioRED")
# ds['test'].to_json('bioRED_test.json')

# from datasets import load_dataset
#
# ds = load_dataset("clinicalnlplab/chemprot_test")
# ds['train'].to_json('chemprot_test.json')
from datasets import load_dataset

ds = load_dataset("kroshan/BioASQ")
ds['train'].to_json('BioASQ_test.json')
