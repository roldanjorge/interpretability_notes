# %% Setup imports and device
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

from src.arena3.ch1_1_transformer_from_scratch.ch2_clean_transformer_implementation.config import Config
from src.arena3.utils.device import device


MAIN = __name__ == "__main__"

# %% 
# Explore the GPT-2 tokenizer
reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,  # you'll learn about these arguments later!
)

sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n: n[1])



# %%
# ======================================
#  Text generation 
# ======================================
# Step 1: Convert text to tokens
reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text).to(device)
print(tokens)
print(tokens.shape)
print(reference_gpt2.to_str_tokens(tokens))

# %% 
# Step 2: Map tokens to logits
logits, cache = reference_gpt2.run_with_cache(tokens)
print(logits.shape)


# %% 
# Step 3: Convert the logits to a distribution with a softmax
probs = logits.softmax(dim=-1)
print(probs.shape)

# %%
# Print All Activation Shapes of Reference Model
# ==============================
for activation_name, activation in cache.items():
    # Only print for first layer
    if ".0." in activation_name or "blocks" not in activation_name:
        print(f"{activation_name:30} {tuple(activation.shape)}")

# %%
# Print All Parameters Shapes of Reference Model
for name, param in reference_gpt2.named_parameters():
    # Only print for first layer
    if ".0." in name or "blocks" not in name:
        print(f"{name:18} {tuple(param.shape)}")

# %% 
# Config
# ==========================================
print(reference_gpt2.cfg)

# %%
# Custom config
cfg = Config()
print(cfg)
# %%
