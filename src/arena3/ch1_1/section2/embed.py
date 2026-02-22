from jaxtyping import Float, Int
import torch as t
from torch import Tensor
import torch.nn as nn

from src.arena3.ch1_1.section2.config import Config

cfg = Config()


class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        tokens_embed = nn.functional.one_hot(tokens, num_classes=self.cfg.d_vocab).to(t.float32)
        out = t.matmul(tokens_embed, self.W_E)
        return out
