# %%
# Setup imports
import circuitsvis as cv
from IPython.display import display
import torch as t
from transformer_lens import HookedTransformer, HookedTransformerConfig

from src.arena3.utils.device import device

# %%
# Setup config
cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True,  # defaults to False
    tokenizer_name="EleutherAI/gpt-neox-20b",
    seed=398,
    use_attn_result=True,
    normalization_type=None,  # defaults to "LN", i.e. layernorm with weights & biases
    positional_embedding_type="shortformer",
)

# %%
# Download models from Huggin Face
from huggingface_hub import hf_hub_download

REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"

weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

# %%
# Instantiate model
model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(pretrained_weights)

# %%
# Exercise - visualise & inspect attention patterns
text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

logits, cache = model.run_with_cache(text, remove_batch_dim=True)

# %%
# Solution
attention_pattern_0 = cache["pattern", 0]
print(attention_pattern_0.shape)
tokens = model.to_str_tokens(text)

print("Layer 0 Head Attention Patterns:")
display(
    cv.attention.attention_patterns(
        tokens=tokens,
        attention=attention_pattern_0,
    )
)

# %%
# jr_solution
# attention_pattern_1 = cache["pattern", 1]
# print(attention_pattern_1.shape)

# print("Layer 1 Head Attention Patterns:")
# display(
#     cv.attention.attention_patterns(
#         tokens=tokens,
#         attention=attention_pattern_1,
#     )
# )

# %%
# reference_solution
str_tokens = model.to_str_tokens(text)
for layer in range(model.cfg.n_layers):
    attention_pattern = cache["pattern", layer]
    display(cv.attention.attention_patterns(tokens=str_tokens, attention=attention_pattern))


# %%
#
