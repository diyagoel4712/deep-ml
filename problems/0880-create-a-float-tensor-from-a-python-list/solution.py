import torch

def to_float_tensor(values):
    tensor=torch.tensor(values, dtype=torch.float32)
    return tensor
