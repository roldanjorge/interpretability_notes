import torch as t
import torch.nn as nn
from torch import Tensor
import einops
from transformer_lens import HookedTransformer
from jaxtyping import Float, Int
from src.config import Config
from tests.mlp_tests import rand_float_test, load_gpt2_test
from transformer_lens.utils import gelu_new

class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        out = (
            einops.einsum(
                normalized_resid_mid,
                self.W_in,
                "batch posn d_model, d_model d_mlp -> batch posn d_mlp"
            )
            + self.b_in
        )
        out = gelu_new(out)
        out = (
            einops.einsum(
                out,
                self.W_out,
                "batch posn d_mlp, d_mlp d_model -> batch posn d_model"
            )
            + self.b_out
        )
        return out


if __name__ == "__main__":
    reference_gpt2 = HookedTransformer.from_pretrained(
        "gpt2-small",
        fold_ln=False,
        center_unembed=False,
        center_writing_weights=False,
    )

    device = t.device(
        "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
    )

    # %% Step 1: Convert text to tokens
    reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
    tokens = reference_gpt2.to_tokens(reference_text).to(device)
    print(tokens)
    print(tokens.shape)
    print(reference_gpt2.to_str_tokens(tokens))

    # %% Step 2: Map tokens to logits
    logits, cache = reference_gpt2.run_with_cache(tokens)
    print(logits.shape)

    # %% Step 3: Convert the logits to a distribution with a softmax
    probs = logits.softmax(dim=-1)
    print(probs.shape)

    # Tests
    rand_float_test(MLP, [2, 4, 768])
    load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["normalized", 0, "ln2"])