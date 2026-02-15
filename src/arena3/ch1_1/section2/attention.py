# %%
from dataclasses import dataclass
import einops
import torch as t
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from src.arena3.ch1_1.section2.config import Config
from src.arena3.utils.device import device


cfg = Config()

class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device))

    # JR Solution
    def forward(
        self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # Calculate query, key, and value vectors
        q = (
            einops.einsum(
                normalized_resid_pre, 
                self.W_Q, 
                "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head"
            ) 
            + self.b_Q 
        )
        k = (
            einops.einsum(
                normalized_resid_pre,
                self.W_K,
                "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head"
            )
            + self.b_K
        )
        v = (
            einops.einsum(
                normalized_resid_pre,
                self.W_V,
                "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head"
            )
            + self.b_V
        )

        # Calculate attention scores, then scale and mask, and apply softmax to get probabilities
        attn_scores = einops.einsum(
            q,
            k,
            "batch posn_q n_heads d_head, batch posn_k n_heads d_head -> batch n_heads posn_q posn_k",
        )
        scaled_attn_scores = attn_scores / self.cfg.d_head**0.5
        masked_attn_scores = self.apply_causal_mask(scaled_attn_scores)
        attn_pattern = masked_attn_scores.softmax(dim=-1)

        # Take weighted sum of value vectors, according to attention probabilities
        z = einops.einsum(
            v,
            attn_pattern,
            "batch posn_k n_heads d_head, batch n_heads posn_q posn_k -> batch posn_q n_heads d_head",
        )

        # Calculate output (by applying matrix W_O and summing over heads, then adding bias b_O)
        attn_out = (
                einops.einsum(
                z,
                self.W_O,
                "batch posn_q n_heads d_head, n_heads d_head d_model -> batch posn_q d_model"
            )
            + self.b_O
        )

        return attn_out


    def apply_causal_mask(
            self,
            attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"],
        ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
            """
            Applies a causal mask to attention scores, and returns masked scores.
            """
            # Define a mask that is True for all positions we want to set probabilities to zero for
            all_ones = t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device)
            mask = t.triu(all_ones, diagonal=1).bool()
            # Apply the mask to attention scores, then return the masked scores
            attn_scores.masked_fill_(mask, self.IGNORE)
            return attn_scores