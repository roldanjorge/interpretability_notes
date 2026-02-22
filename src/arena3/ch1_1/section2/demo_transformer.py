from jaxtyping import Float, Int
from torch import Tensor
import torch.nn as nn

from src.arena3.ch1_1.section2.config import Config
from src.arena3.ch1_1.section2.embed import Embed
from src.arena3.ch1_1.section2.layer_norm import LayerNorm
from src.arena3.ch1_1.section2.pos_embed import PosEmbed
from src.arena3.ch1_1.section2.transformer_block import TransformerBlock
from src.arena3.ch1_1.section2.unembed import Unembed

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

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        x = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        out = self.unembed(x)
        return out  # type: ignore[no-any-return]
