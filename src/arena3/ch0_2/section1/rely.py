# %%
# imports
import torch as t
from torch import Tensor
import torch.nn as nn


# %%
# ReLU implementation
class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return t.maximum(x, t.tensor(0.0))
