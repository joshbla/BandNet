# Download CUDA version from archive: https://developer.nvidia.com/cuda-toolkit-archive
# Download cuDNN version from archive: https://developer.nvidia.com/rdp/cudnn-archive
# pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

import torch

# Check PyTorch version
print(torch.version.cuda)

# Check CUDA availability
print(torch.cuda.is_available())

# Check cuDNN availability
print(torch.backends.cudnn.is_available())

# Get the current cuDNN version (if available)
print(torch.backends.cudnn.version())
