#!/bin/bash
set -e  # 遇到错误立即退出

modelscope upload MasterDu/qwen3dr-8b-kto model/ckpt-3000-kto --token ms-86c9f700-3bb5-4cdf-8206-dc7322d584fc
modelscope upload MasterDu/qwen3dr-8b-kto-ckpt-500 model/checkpoint-500 --token ms-86c9f700-3bb5-4cdf-8206-dc7322d584fc
modelscope upload MasterDu/qwen3dr-8b-kto-ckpt-1000 model/checkpoint-1000 --token ms-86c9f700-3bb5-4cdf-8206-dc7322d584fc
modelscope upload MasterDu/qwen3dr-8b-kto-ckpt-1500 model/checkpoint-1500 --token ms-86c9f700-3bb5-4cdf-8206-dc7322d584fc
modelscope upload MasterDu/qwen3dr-8b-kto-ckpt-2000 model/checkpoint-2000 --token ms-86c9f700-3bb5-4cdf-8206-dc7322d584fc
modelscope upload MasterDu/qwen3dr-8b-kto-ckpt-2500 model/checkpoint-2500 --token ms-86c9f700-3bb5-4cdf-8206-dc7322d584fc
modelscope upload MasterDu/qwen3dr-8b-kto-ckpt-3000 model/checkpoint-3000 --token ms-86c9f700-3bb5-4cdf-8206-dc7322d584fc
modelscope upload MasterDu/qwen3dr-8b-kto-ckpt-3500 model/checkpoint-3500 --token ms-86c9f700-3bb5-4cdf-8206-dc7322d584fc
modelscope upload MasterDu/qwen3dr-8b-kto-ckpt-4000 model/checkpoint-4000 --token ms-86c9f700-3bb5-4cdf-8206-dc7322d584fc
