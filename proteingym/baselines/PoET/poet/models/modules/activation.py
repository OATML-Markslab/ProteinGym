import torch
import torch.nn.functional as F

gelu = F.gelu

# def gelu(x):
#    return x*0.5*(1.0 + torch.erf(x/1.41421356237))
