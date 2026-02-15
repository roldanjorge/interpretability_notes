import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from src.arena3.ch1_1_transformer_from_scratch.ch2_clean_transformer_implementation.config import Config
from src.arena3.ch1_1_transformer_from_scratch.ch2_clean_transformer_implementation.layer_norm import LayerNorm
from src.arena3.ch1_1_transformer_from_scratch.ch2_clean_transformer_implementation.attention import Attention
from src.arena3.ch1_1_transformer_from_scratch.ch2_clean_transformer_implementation.mlp import MLP


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