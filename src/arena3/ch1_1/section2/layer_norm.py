# %%
from jaxtyping import Float
import torch as t
from torch import Tensor
import torch.nn as nn

from src.arena3.ch1_1.section2.config import Config

cfg = Config()


class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        """
        Reference:
            PyTorch: https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
            Paper: https://arxiv.org/abs/1607.06450
        """
        u = t.mean(residual, dim=-1, keepdim=True)
        s = t.var(residual, dim=-1, keepdim=True, unbiased=False)
        out = (residual - u) / t.sqrt(s + self.cfg.layer_norm_eps)
        out = self.w * out + self.b
        return out
