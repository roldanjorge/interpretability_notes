import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import datasets
import einops
import numpy as np
import torch as t
import torch.nn as nn
import wandb
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from config import Config

from layer_norm import LayerNorm
from attention import Attention
from mlp import MLP


@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


cfg = Config()


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    # JR Solution
    def forward(
        self, resid_pre: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        pre_attn = self.ln1(resid_pre)
        post_attn = self.attn(pre_attn)
        new_resid_pre = post_attn + resid_pre
        pre_mlp = self.ln2(new_resid_pre)
        post_mlp = self.mlp(pre_mlp)
        out = new_resid_pre + post_mlp
        return out 

    # Reference solution
    # def forward(
    #     self, resid_pre: Float[Tensor, "batch position d_model"]
    # ) -> Float[Tensor, "batch position d_model"]:
    #     resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
    #     resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
    #     return resid_post