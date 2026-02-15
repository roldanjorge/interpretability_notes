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


from src.arena3.ch1_1.section2.embed import Embed
from src.arena3.ch1_1.section2.pos_embed import PosEmbed
from src.arena3.ch1_1.section2.transformer_block import TransformerBlock
from src.arena3.ch1_1.section2.layer_norm import LayerNorm
from src.arena3.ch1_1.section2.unembed import Unembed
from src.arena3.ch1_1.section2.config import Config
from src.arena3.utils.device import device


cfg = Config()


class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    # JR Solution
    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        x = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        out = self.unembed(x)
        return out

    # Reference solution
    # def forward(
    #     self, tokens: Int[Tensor, "batch position"]
    # ) -> Float[Tensor, "batch position d_vocab"]:
    #     residual = self.embed(tokens) + self.pos_embed(tokens)
    #     for block in self.blocks:
    #         residual = block(residual)
    #     logits = self.unembed(self.ln_final(residual))
    #     return logits
