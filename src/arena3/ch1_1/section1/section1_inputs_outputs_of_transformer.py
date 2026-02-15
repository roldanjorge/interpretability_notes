# %% Setup imports and device
import torch as t
from transformer_lens import HookedTransformer
from src.arena3.utils.device import device

# %% 
# ======================================
# Splitting language into sub-units
# ======================================
# %% Explore the GPT-2 tokenizer
reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,  # you'll learn about these arguments later!
)

sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n: n[1])

# %% Print out some tokens from the GPT-2 tokenizer
print(sorted_vocab[:20])
print()
print(sorted_vocab[250:270])
print()
print(sorted_vocab[990:1010])
print()
print(sorted_vocab[-20:])
print()

# %%
# ======================================
# Some tokenization annoyances 
# ======================================
print(reference_gpt2.to_str_tokens("Ralph"))
print(reference_gpt2.to_str_tokens(" Ralph"))
print(reference_gpt2.to_str_tokens(" ralph"))
print(reference_gpt2.to_str_tokens("ralph"))

# %% Arithmetic is a mess
print(reference_gpt2.to_str_tokens("56873+3184623=123456789-1000000000")) 

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
# Bonus step: What is the most likely next token at each position?
most_likely_next_tokens = reference_gpt2.tokenizer.batch_decode(logits.argmax(dim=-1)[0])

print(list(zip(reference_gpt2.to_str_tokens(tokens), most_likely_next_tokens)))

# %% 
# Step 4: Map distribution to a token
next_token = logits[0, -1].argmax(dim=-1)
next_char = reference_gpt2.to_string(next_token)
print(repr(next_char))


# %% 
# Step 5: Add this to the end of the input, re-run
print(f"Sequence so far: {reference_gpt2.to_string(tokens)[0]!r}")

for i in range(10):
    print(f"{tokens.shape[-1] + 1}th char = {next_char!r}")
    # Define new input sequence, by appending the previously generated token
    tokens = t.cat([tokens, next_token[None, None]], dim=-1)
    # Pass our new sequence through the model, to get new output
    logits = reference_gpt2(tokens)
    # Get the predicted token at the end of our sequence
    next_token = logits[0, -1].argmax(dim=-1)
    # Decode and print the result
    next_char = reference_gpt2.to_string(next_token)

# %%
