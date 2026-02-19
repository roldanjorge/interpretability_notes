# %%
# imports
import torch as t
import torch.nn as nn
from torch import Tensor


# %%
# ReLU implementation
class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return t.maximum(x, t.tensor(0.0))