# %%
# Setup imports and device
import math

from tqdm.notebook import tqdm
from transformer_lens import HookedTransformer

from src.arena3.ch1_1.section2.config import Config
from src.arena3.ch1_1.section2.demo_transformer import DemoTransformer
from src.arena3.utils.device import device
from src.arena3.utils.utils import get_log_probs

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

# Demo Transformer
# ==========================================
demo_gpt2 = DemoTransformer(Config(debug=False)).to(device)
demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)
demo_logits = demo_gpt2(tokens)
print(demo_logits)


# %%
# Get log-probs for demo_gpt2
pred_log_probs = get_log_probs(demo_logits, tokens)
print(f"Avg cross entropy loss: {-pred_log_probs.mean():.4f}")
print(f"Avg cross entropy loss for uniform distribution: {math.log(demo_gpt2.cfg.d_vocab):4f}")
print(f"Avg probability assigned to correct token: {pred_log_probs.exp().mean():4f}")

# %%
# Generate text with demo_gpt2
test_string = """Mitigating the risk of extinction from AI should be a global priority alongside other societal-scale risks such as"""
for i in tqdm(range(100)):
    test_tokens = reference_gpt2.to_tokens(test_string).to(device)
    demo_logits = demo_gpt2(test_tokens)
    test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())

print(test_string)
# %%
# Visualize attention patterns with circuitsvis
import circuitsvis as cv
from IPython.display import display

display(
    cv.attention.attention_patterns(
        tokens=reference_gpt2.to_str_tokens(reference_text),
        attention=cache["pattern", 0][0],
    )
)
# %%
display(
    cv.attention.attention_heads(
        tokens=reference_gpt2.to_str_tokens(reference_text),
        attention=cache["pattern", 0][0],
    )
)

# %%
