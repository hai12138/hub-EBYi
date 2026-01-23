# 测试 jieba
import jieba
print("jieba 的版本是：",jieba.__version__)

# 测试 sklearn（注意：安装的是 scikit-learn，但导入用 sklearn）
import sklearn
print("sklearn 的版本是：",sklearn.__version__)

# 测试 PyTorch
import torch
print("torch 的版本是：",torch.__version__)
print(torch.cuda.is_available())  # 检查是否能用 GPU

import numpy
print("numpy 的版本是：",numpy.__version__)

import pandas
print("pandas 的版本是：",pandas.__version__)

