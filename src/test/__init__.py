import torch

# pip.exe uninstall torch torchvision torchaudio -y
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

print(torch.cuda.is_available())