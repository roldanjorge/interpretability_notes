# %%
import einops
import torch as t
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from src.arena3.ch1_1.section2.config import Config

cfg = Config()

class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        batch, seq_len = tokens.shape
        out = self.W_pos[:seq_len]
        out = einops.repeat(out, "seq d_model -> batch seq d_model", batch=batch)
        return out