import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
# 使用函数方式下载
model_dir = snapshot_download('ZhipuAI/glm-4-9b-chat-1M', cache_dir='/home/data/GLM-4/modelTemp', revision='master')