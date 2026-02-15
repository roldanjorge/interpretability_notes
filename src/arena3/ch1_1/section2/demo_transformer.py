# %%
import math
from tqdm.notebook import tqdm
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer


from src.arena3.ch1_1.section2.embed import Embed
from src.arena3.ch1_1.section2.pos_embed import PosEmbed
from src.arena3.ch1_1.section2.transformer_block import TransformerBlock
from src.arena3.ch1_1.section2.layer_norm import LayerNorm
from src.arena3.ch1_1.section2.unembed import Unembed
from src.arena3.ch1_1.section2.config import Config
from src.arena3.utils.device import device


# %%
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

# %%
def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = (
        log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    )

    return log_probs_for_tokens

# %%
if __name__ == "__main__":
    reference_gpt2 = HookedTransformer.from_pretrained(
        "gpt2-small",
        fold_ln=False,
        center_unembed=False,
        center_writing_weights=False,
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

    # %% Try demo_gpt2
    demo_gpt2 = DemoTransformer(Config(debug=False)).to(device)
    demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)
    demo_logits = demo_gpt2(tokens)
    print(demo_logits)

    # %%
    pred_log_probs = get_log_probs(demo_logits, tokens)
    print(f"Avg cross entropy loss: {-pred_log_probs.mean():.4f}")
    print(f"Avg cross entropy loss for uniform distribution: {math.log(demo_gpt2.cfg.d_vocab):4f}")
    print(f"Avg probability assigned to correct token: {pred_log_probs.exp().mean():4f}")

    # %%
    test_string = """Mitigating the risk of extinction from AI should be a global priority alongside other societal-scale risks such as"""
    for i in tqdm(range(100)):
        test_tokens = reference_gpt2.to_tokens(test_string).to(device)
        demo_logits = demo_gpt2(test_tokens)
        test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())

    print(test_string)


# %%
