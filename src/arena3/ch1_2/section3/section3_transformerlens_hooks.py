# %%
# Setup imports
import circuitsvis as cv
import einops
from IPython.display import display
from jaxtyping import Float, Int
import torch as t
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from src.arena3.ch1_2.plotly_utils import imshow
from src.arena3.ch1_2.section2.section2_finding_induction_heads import cfg, generate_repeated_tokens
from src.arena3.utils.device import device

# %%
# Setup MAIN
MAIN = __name__ == "__main__"

# %%
# Instantiate model
if MAIN:
    from huggingface_hub import hf_hub_download

    REPO_ID = "callummcdougall/attn_only_2L_half"
    FILENAME = "attn_only_2L_half.pth"

    weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

    model = HookedTransformer(cfg)
    pretrained_weights = t.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(pretrained_weights)


# %%
# Exercise - calculate induction scores with hooks
seq_len = 50
batch_size = 10
rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch_size)

# We make a tensor to store the induction score for each head.
# We put it on the model's device to avoid needing to move things between the GPU and CPU,
# which can be slow.
induction_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)


# %%
def induction_score_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"], hook: HookPoint
):
    """
    Calculates the induction score, and stores it in the [layer, head] position of the
    `induction_score_store` tensor.
    """
    # Take the diagonal of attn paid from each dest posn to src posns (seq_len-1) tokens back
    # (This only has entries for tokens with index>=seq_len)
    induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1 - seq_len)

    # Get an average score per head
    induction_score = einops.reduce(
        induction_stripe, "batch head_index position -> head_index", "mean"
    )

    # Store result
    induction_score_store[hook.layer(), :] = induction_score


# %%
if MAIN:
    # We make a boolean filter on activation names, that's true only on attention pattern names
    pattern_hook_names_filter = lambda name: name.endswith("pattern")

    # Run with hooks (this is where we write to the `induction_score_store` tensor`)
    model.run_with_hooks(
        rep_tokens_10,
        return_type=None,  # For efficiency, we don't need to calculate the logits
        fwd_hooks=[(pattern_hook_names_filter, induction_score_hook)],
    )

    # Plot the induction scores for each head in each layer
    imshow(
        induction_score_store,
        labels={"x": "Head", "y": "Layer"},
        title="Induction Score by Head",
        text_auto=".2f",
        width=900,
        height=350,
    )


# %%
# Exercise - find induction heads in GPT2-small
def visualize_pattern_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    print("Layer: ", hook.layer())
    display(
        cv.attention.attention_patterns(
            tokens=gpt2_small.to_str_tokens(rep_tokens_10[0]), attention=pattern.mean(0)
        )
    )


# YOUR CODE HERE - find induction heads in gpt2_small

# %%
# Visualize all 12 layers of the gpt2_small model
if MAIN:
    rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch_size)

    gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
    config = gpt2_small.cfg
    print(config)

    induction_score_store = t.zeros(
        (gpt2_small.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device
    )

    gpt2_small.run_with_hooks(
        rep_tokens_10,
        return_type=None,
        fwd_hooks=[
            (pattern_hook_names_filter, visualize_pattern_hook),
            (pattern_hook_names_filter, induction_score_hook),
        ],
    )

    # Plot the induction scores for each head in each layer
    imshow(
        induction_score_store,
        labels={"x": "Head", "y": "Layer"},
        title="Induction Score by Head",
        text_auto=".2f",
        width=900,
        height=350,
    )


# %%
# Exercise - build logit attribution tool
def logit_attribution(
    embed: Float[Tensor, "seq d_model"],
    l1_results: Float[Tensor, "seq nheads d_model"],
    l2_results: Float[Tensor, "seq nheads d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    tokens: Int[Tensor, "seq"],
) -> Float[Tensor, "seq-1 n_components"]:
    """
    Inputs:
        embed: the embeddings of the tokens (i.e. token + position embeddings)
        l1_results: the outputs of the attention heads at layer 1 (with head as one of the dims)
        l2_results: the outputs of the attention heads at layer 2 (with head as one of the dims)
        W_U: the unembedding matrix
        tokens: the token ids of the sequence

    Returns:
        Tensor of shape (seq_len-1, n_components)
        represents the concatenation (along dim=-1) of logit attributions from:
            the direct path (seq-1,1)
            layer 0 logits (seq-1, n_heads)
            layer 1 logits (seq-1, n_heads)
        so n_components = 1 + 2*n_heads
    """
    W_U_correct_tokens = W_U[:, tokens[1:]]

    # jr_solution
    # output_0_col = einops.einsum(embed[:-1], W_U_correct_tokens, "seq d_model, d_model seq -> seq").unsqueeze(dim=-1)
    # output_layer0 = einops.einsum(l1_results[:-1], W_U_correct_tokens, "seq nheads d_model, d_model seq -> seq nheads")
    # output_layer1 = einops.einsum(l2_results[:-1], W_U_correct_tokens, "seq nheads d_model, d_model seq -> seq nheads")
    # output = t.concat([output_0_col, output_layer0, output_layer1], dim=-1)
    # return output

    # reference_solution
    direct_attributions = einops.einsum(W_U_correct_tokens, embed[:-1], "emb seq, seq emb -> seq")
    l1_attributions = einops.einsum(
        W_U_correct_tokens, l1_results[:-1], "emb seq, seq nhead emb -> seq nhead"
    )
    l2_attributions = einops.einsum(
        W_U_correct_tokens, l2_results[:-1], "emb seq, seq nhead emb -> seq nhead"
    )
    return t.concat([direct_attributions.unsqueeze(-1), l1_attributions, l2_attributions], dim=-1)


# %%
if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
    logits, cache = model.run_with_cache(text, remove_batch_dim=True)
    str_tokens = model.to_str_tokens(text)
    tokens = model.to_tokens(text)

    with t.inference_mode():
        embed = cache["embed"]
        l1_results = cache["result", 0]
        l2_results = cache["result", 1]
        logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
        # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
        correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
        t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
        print("Tests passed!")

# %%
