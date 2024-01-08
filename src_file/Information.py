# %%
import sys
import torch

#バージョンの確認Python
print("\n-バージョンの確認Python")
print(f"Python version {sys.version}")

#Pytorch・CUDA確認
#URL：https://qiita.com/Haaamaaaaa/items/d237456302ef9332f6e4
print("\n-Pytorch・CUDA確認")
print(torch.__version__)
print(f"cuda, {torch.cuda.is_available()}")
print(f"compute_{''.join(map(str,(torch.cuda.get_device_capability())))}")
device_num:int = torch.cuda.device_count()
print(f"find gpu devices, {device_num}")
for idx in range(device_num):
    print(f"cuda:{idx}, {torch.cuda.get_device_name(idx)}")

#動作
print("\n-動作")
x = torch.rand(5, 3)
print(x)

