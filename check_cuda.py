import torch
print(f"CUDA installed? {torch.cuda.is_available()}")

torch.Tensor([1, 2, 3]).to("cuda")