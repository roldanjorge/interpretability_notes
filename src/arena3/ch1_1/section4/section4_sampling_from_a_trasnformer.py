# %%

import torch as t
from transformer_lens import HookedTransformer

from src.arena3.ch1_1.section2.config import Config
from src.arena3.ch1_1.section2.demo_transformer import DemoTransformer
from src.arena3.ch1_1.section4.transformer_sampler import TransformerSampler
from src.arena3.utils.device import device

# %%
# Setup reference model
reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,  # you'll learn about these arguments later!
)


# %%
t.set_grad_enabled(False)  # gradients are not necessary for sampling

model = DemoTransformer(Config()).to(device)
model.load_state_dict(reference_gpt2.state_dict(), strict=False)
tokenizer = reference_gpt2.tokenizer
sampler = TransformerSampler(model, tokenizer)

prompt = "Jingle bells, jingle bells, jingle all the way"
print(f"Testing greedy decoding\nPrompt:   {prompt!r}")

expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
output = sampler.sample(prompt, max_tokens_generated=8, temperature=0.0)

print(f"Expected: {expected!r}\nActual:   {output!r}\n")
assert output == expected

print("Tests passed!")

# %%
