# pip install "torch>=2.5.1" "torchvision>=0.20.1" "torchaudio>=2.5.1" --index-url https://download.pytorch.org/whl/cu126

import torch
print(torch.cuda.is_available())
