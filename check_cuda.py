import torch
print(f"CUDA installed? {torch.cuda.is_available()}")
available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print(f"Available GPUs: {available_gpus}")
torch.Tensor([1, 2, 3]).to("cuda")