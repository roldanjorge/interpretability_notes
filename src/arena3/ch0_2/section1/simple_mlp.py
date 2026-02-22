# %%
# imports
from torch import Tensor
import torch.nn as nn

from src.arena3.ch0_2.section1.flatten import Flatten
from src.arena3.ch0_2.section1.linear import Linear
from src.arena3.ch0_2.section1.rely import ReLU


# %%
# SimpleMLP implementation
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.linear1 = Linear(in_features=28 * 28, out_features=100)
        self.relu = ReLU()
        self.linear2 = Linear(in_features=100, out_features=10)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.relu(self.linear1(self.flatten(x))))  # type: ignore[no-any-return]
