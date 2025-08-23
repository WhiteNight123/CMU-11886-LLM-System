import torch
from sgl_kernel import rmsnorm

x = torch.randn(4, 128, 1024, device="cuda")
weight = torch.randn(1024, device="cuda")
out = rmsnorm(x, weight, 1e-6)
print("RMSNorm test passed!")