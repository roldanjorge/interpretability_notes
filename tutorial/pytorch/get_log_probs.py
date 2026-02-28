# get_log_probs.py
# Tutorial: understanding get_log_probs step by step using PyTorch primitives.
#
# Every #%% marker is a runnable cell (VS Code, Spyder, PyCharm).
# Run cells top to bottom; each one builds on the tensors defined above it.

# %% ── 0. Imports ──────────────────────────────────────────────────────────────

import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)

# %% ── 1. The function we are explaining ──────────────────────────────────────
#
# Given:
#   logits : Float[batch, posn, d_vocab]   raw scores from the transformer
#   tokens : Int  [batch, posn]            actual token IDs in the sequence
#
# Returns:
#   Float[batch, posn-1]   log P(correct next token) at every position


def get_log_probs(logits, tokens):
    log_probs = logits.log_softmax(dim=-1)  # step 1
    log_probs_for_tokens = (
        log_probs[:, :-1]  # step 2
        .gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1))  # step 3 + 4
        .squeeze(-1)  # step 5
    )
    return log_probs_for_tokens


# %% ── 2. Build a toy example ─────────────────────────────────────────────────
#
# Sequence: ["The", "cat", "sat", "."]   token IDs: [1, 2, 3, 4]
# 1 batch item, 4 positions, tiny vocab of 6 tokens.

BATCH = 1
POSN = 4  # sequence length
D_VOCAB = 6  # vocabulary size

# Simulate logits: random unnormalised scores the model might produce.
logits = torch.randn(BATCH, POSN, D_VOCAB)

# Ground-truth token IDs.
#   pos 0 → "The"  (1)
#   pos 1 → "cat"  (2)
#   pos 2 → "sat"  (3)
#   pos 3 → "."    (4)
tokens = torch.tensor([[1, 2, 3, 4]])  # shape [1, 4]

print("logits shape :", logits.shape)  # [1, 4, 6]
print("tokens shape :", tokens.shape)  # [1, 4]
print()
print("logits:\n", logits)
print()
print("tokens:\n", tokens)

# %% ── 3. Step 1 — log_softmax along the vocab dimension ─────────────────────
#
# logits[b, p, :] is a vector of D_VOCAB raw scores for position p in batch b.
# The scores are unnormalised — they can be any real number and do not sum to 1.
#
# ── From a single logit vector to a probability distribution ─────────────────
#
# Softmax first turns the raw scores into probabilities:
#
#   P(token_v | context) = exp(logits[b, p, v])
#                          ─────────────────────────────────────────
#                          sum over all v' of exp(logits[b, p, v'])
#
#   The denominator sums exp(score) across the entire vocabulary,
#   so every entry is non-negative and the whole vector sums to 1.
#
# Taking the log of both sides gives log-softmax:
#
#   log P(token_v | context) = log(  exp(logits[b, p, v])
#                                    ─────────────────────────────── )
#                                    sum_v' exp(logits[b, p, v'])
#
#                            = log exp(logits[b, p, v])
#                              - log( sum_v' exp(logits[b, p, v']) )
#
#                            = logits[b, p, v]
#                              - log( sum_v' exp(logits[b, p, v']) )
#
#   The second step uses  log(a/b) = log(a) - log(b).
#   The third step uses   log(exp(x)) = x.
#   The subtracted term is a single scalar (the log-normaliser) that is the
#   same for every v — it does not depend on v, only on the full score vector
#   at position p for batch item b.
#
# ── What "context up to position p" means ───────────────────────────────────
#
#   A causal transformer at position p has seen tokens at positions 0 … p
#   (it cannot attend to future positions).  Its hidden state at position p
#   therefore encodes the entire left context, and the logit vector
#   logits[b, p, :] is the model's output distribution *given* that context:
#   "what token is most likely to come next?".
#
#   So  log_probs[b, p, v]  = log P(next token = v | tokens[b, 0], …, tokens[b, p])
#
# ── dim=-1 ───────────────────────────────────────────────────────────────────
#
#   The normalising sum runs over all vocabulary entries, which live on the
#   last axis (dim=-1, size D_VOCAB).  Using dim=-1 applies the operation
#   independently for every (b, p) pair — i.e. one separate softmax per
#   position per batch item — without touching the batch or position axes.
#   The shape is unchanged: [batch, posn, d_vocab].
#
# ── Why log-softmax instead of softmax + log ────────────────────────────────
#
#   For likely next tokens the raw probability can be very close to 1, but
#   for unlikely tokens it can be so small (e.g. 1e-40) that float32 rounds
#   it to 0 before the log is taken, producing -inf.  log_softmax computes
#   the same result algebraically without ever materialising tiny probabilities,
#   so it is numerically stable.

log_probs = logits.log_softmax(dim=-1)

print("log_probs shape:", log_probs.shape)  # still [1, 4, 6]

# ── Verify the formula by hand for position 0, batch item 0 ─────────────────
b, p = 0, 0
score_vec = logits[b, p]  # raw scores, shape [D_VOCAB]
log_normaliser = score_vec.exp().sum().log()  # log( sum_v' exp(score_v') )
log_probs_manual = score_vec - log_normaliser  # broadcast: one scalar subtracted

print("\nPosition p=0, batch b=0")
print("  raw logits          :", score_vec)
print("  log-normaliser      :", log_normaliser.item())
print("  manual log_probs    :", log_probs_manual)
print("  torch  log_probs    :", log_probs[b, p])
assert torch.allclose(log_probs_manual, log_probs[b, p]), "formula mismatch"
print("  formula matches torch ✓")

# ── The log-normaliser is the same scalar for every v ───────────────────────
# Subtracting it shifts the entire score vector down by the same amount so
# that exp(result) sums to 1 across v — it is the normalisation constant.
print("\n  exp(log_probs[b,p]).sum() [should be 1.0]:", log_probs[b, p].exp().sum().item())

# ── Sanity check: must hold for every (b, p) pair ───────────────────────────
prob_sums = log_probs.exp().sum(dim=-1)
print("\nexp(log_probs).sum(dim=-1) [should be all 1.0]:\n", prob_sums)

# %% ── 4. Step 2 — log_probs[:, :-1] — slicing a 3-D tensor ──────────────────
#
# SHAPE BEFORE: [batch, posn,   d_vocab]  =  [1, 4, 6]
# SHAPE AFTER : [batch, posn-1, d_vocab]  =  [1, 3, 6]
#
# The index expression  [:, :-1]  contains only TWO slots, but the tensor
# has THREE dimensions.  PyTorch (like NumPy) applies a simple rule:
#
#   Any trailing dimension not mentioned gets an implicit ':'
#   (meaning "keep every element along that axis").
#
# So  [:, :-1]  is identical to  [:, :-1, :]  — explicitly written:
#
#   dim 0  →  :      keep all batch items                (unchanged)
#   dim 1  →  :-1    keep positions 0 … posn-2, DROP the last position
#   dim 2  →  :      keep ALL vocabulary entries         (implicit, unchanged)
#
# Think of log_probs as a stack of (batch) matrices each of shape
# [posn x d_vocab].  The slice removes only the LAST ROW of each matrix
# while leaving every row's full width (d_vocab columns) completely intact.
#
# Why drop the last position?
#   The transformer at position (posn-1) produces a prediction, but there is
#   no ground-truth *next* token to compare it against, so it cannot contribute
#   to the loss and is simply discarded here.

lp_sliced = log_probs[:, :-1]  # identical to log_probs[:, :-1, :]

print("log_probs shape  :", log_probs.shape)  # [1, 4, 6]
print("lp_sliced shape  :", lp_sliced.shape)  # [1, 3, 6]
print()

# Show explicitly that [:, :-1] and [:, :-1, :] are the same tensor.
assert torch.equal(log_probs[:, :-1], log_probs[:, :-1, :]), "should be identical"
print("log_probs[:, :-1] == log_probs[:, :-1, :] ✓")
print()

# Inspect: which rows were kept, which was dropped?
print("log_probs (all positions):\n", log_probs[0])  # 4 rows x 6 cols
print()
print("log_probs[:, :-1] (positions 0-2 only):\n", lp_sliced[0])  # 3 rows x 6 cols
print()
print("Dropped row (position 3):\n", log_probs[0, -1])  # the row that was removed

# %% ── 5. Step 3 — tokens[:, 1:] — slicing a 2-D tensor ─────────────────────
#
# SHAPE BEFORE: [batch, posn]    =  [1, 4]
# SHAPE AFTER : [batch, posn-1]  =  [1, 3]
#
# Index expression  [:, 1:]  maps to:
#
#   dim 0  →  :    keep all batch items
#   dim 1  →  1:   keep indices 1, 2, …, posn-1  — DROP index 0
#
# General Python slice  a:b  keeps every index i where  a <= i < b.
# Omitting b means "go to the end of the axis".
# So  1:  keeps {1, 2, 3, …} and discards only index 0 (the first token).
#
# Effect: LEFT-SHIFT by one position.
#   result[b, i]  ==  tokens[b, i+1]   for every b, i
#
#   original tokens:  ["The"(1), "cat"(2), "sat"(3), "."(4)]
#   after  [:, 1:]:               ["cat"(2), "sat"(3), "."(4)]
#                                    ↑ index 0   ↑ index 1   ↑ index 2
#
# Why drop the first token?
#   "The" (index 0) is the very first token — no model position exists that
#   should have predicted it as a *next* token, so it has no role as a target.
#   The remaining tokens are the ground-truth TARGETS:
#     - tokens[:, 1:][b, 0] = "cat"  → what position 0 ("The") should predict
#     - tokens[:, 1:][b, 1] = "sat"  → what position 1 ("cat") should predict
#     - tokens[:, 1:][b, 2] = "."    → what position 2 ("sat") should predict

tok_sliced = tokens[:, 1:]

print("tokens shape    :", tokens.shape)  # [1, 4]
print("tok_sliced shape:", tok_sliced.shape)  # [1, 3]
print()
print("tokens    :", tokens)  # [[1, 2, 3, 4]]
print("tok_sliced:", tok_sliced)  # [[   2, 3, 4]]

# %% ── 6. Alignment: both slices now share the same [batch, posn-1] prefix ───
#
# After the two slices:
#
#   lp_sliced   shape [batch, posn-1, d_vocab]  = [1, 3, 6]
#   tok_sliced  shape [batch, posn-1]            = [1, 3]
#
# Position i in lp_sliced  → the distribution the model output at step i
# Position i in tok_sliced → the token the model SHOULD have predicted at step i
#
#   lp_sliced [b, 0, :]  ←→  tok_sliced[b, 0]  ("cat")   target for pos 0
#   lp_sliced [b, 1, :]  ←→  tok_sliced[b, 1]  ("sat")   target for pos 1
#   lp_sliced [b, 2, :]  ←→  tok_sliced[b, 2]  (".")     target for pos 2
#
# The last log_prob row (pos 3) and the first token (index 0) were each
# discarded — they are the unmatched ends with no counterpart.

for i, (pos_name, tgt_name) in enumerate([(0, "cat"), (1, "sat"), (2, ".")]):
    print(
        f"  pos {pos_name}: distribution over {D_VOCAB} tokens → "
        f"target = '{tgt_name}' (id={tok_sliced[0, i].item()})"
    )

# %% ── 7. Step 4 — unsqueeze(-1): add trailing dim for gather ─────────────────
#
# torch.gather(input, dim, index) requires:
#   index.shape == input.shape   (same number of dimensions)
#
# lp_sliced  has 3 dims: [batch, posn-1, d_vocab]
# tok_sliced has 2 dims: [batch, posn-1]
#
# unsqueeze(-1) inserts a size-1 dimension at the last axis:
#   [batch, posn-1]  →  [batch, posn-1, 1]
#
# The value 1 means "we want to pick exactly one entry along the vocab axis
# for every (batch, posn) slot" — which is exactly what gather will do.

idx = tok_sliced.unsqueeze(-1)

print("tok_sliced shape:", tok_sliced.shape)  # [1, 3]
print("idx shape       :", idx.shape)  # [1, 3, 1]
print()
print("idx:\n", idx)

# %% ── 8. Step 5 — gather: pick one vocab entry per (batch, position) slot ───
#
# gather(dim=-1, index=idx) reads, for each coordinate (b, p, 0):
#
#   output[b, p, 0] = lp_sliced[b, p, idx[b, p, 0]]
#                   = lp_sliced[b, p, tokens[b, p+1]]
#                   = log P(true next token | context up to position p)
#
# Concretely for batch item 0:
#   output[0, 0, 0] = lp_sliced[0, 0, 2]   ← log P("cat" | "The")
#   output[0, 1, 0] = lp_sliced[0, 1, 3]   ← log P("sat" | "The cat")
#   output[0, 2, 0] = lp_sliced[0, 2, 4]   ← log P("."   | "The cat sat")
#
# Shape after gather: [batch, posn-1, 1]  (the vocab axis collapsed to size 1)

gathered = lp_sliced.gather(dim=-1, index=idx)

print("lp_sliced shape:", lp_sliced.shape)  # [1, 3, 6]
print("gathered shape :", gathered.shape)  # [1, 3, 1]
print()
print("gathered:\n", gathered)

# Verify manually: each value should equal the entry at the target token index.
for p in range(POSN - 1):
    target_id = tok_sliced[0, p].item()
    manual_val = lp_sliced[0, p, target_id].item()
    gather_val = gathered[0, p, 0].item()
    print(
        f"  pos {p}: target id={target_id}  "
        f"manual={manual_val:.4f}  gather={gather_val:.4f}  match={manual_val == gather_val}"
    )

# %% ── 9. Step 6 — squeeze(-1): remove the trailing size-1 dimension ──────────
#
# gathered has shape [batch, posn-1, 1].
# The trailing 1 was only there to satisfy gather's dimension requirement.
# squeeze(-1) removes it, yielding the clean output shape [batch, posn-1].

result = gathered.squeeze(-1)

print("gathered shape:", gathered.shape)  # [1, 3, 1]
print("result shape  :", result.shape)  # [1, 3]
print()
print("result:", result)
print()
print("Interpretation:")
print(f"  log P('cat' | 'The')         = {result[0, 0].item():.4f}")
print(f"  log P('sat' | 'The cat')     = {result[0, 1].item():.4f}")
print(f"  log P('.'   | 'The cat sat') = {result[0, 2].item():.4f}")

# %% ── 10. End-to-end: call the function and compare ─────────────────────────

output = get_log_probs(logits, tokens)

print("get_log_probs output:", output)
print("manual result       :", result)
print()
assert torch.allclose(output, result), "mismatch!"
print("Both approaches give identical results ✓")

# %% ── 11. Batch dimension: verify it works with batch > 1 ───────────────────
#
# Nothing in the logic is specific to a single sequence.
# Every slice and gather operates independently along dim 0.

BATCH2 = 3
logits2 = torch.randn(BATCH2, POSN, D_VOCAB)
tokens2 = torch.randint(0, D_VOCAB, (BATCH2, POSN))

output2 = get_log_probs(logits2, tokens2)

print("logits2 shape :", logits2.shape)  # [3, 4, 6]
print("tokens2 shape :", tokens2.shape)  # [3, 4]
print("output2 shape :", output2.shape)  # [3, 3]  ← batch=3, posn-1=3
print()
print("output2:\n", output2)

# Each row is one batch item; each column is one prediction position.
# output2[b, p] = log P(tokens2[b, p+1] | tokens2[b, :p+1])

# %% ── 12. Connection to cross-entropy loss ───────────────────────────────────
#
# The standard language-modelling loss is the mean of the negated log probs:
#
#   L = - (1 / (posn-1)) * sum_p  log P(x_{p+1} | x_{<=p})
#
# That is just -mean(get_log_probs(logits, tokens)).

log_probs_per_token = get_log_probs(logits2, tokens2)  # [3, 3]

# Mean over all positions and all batch items.
loss_manual = -log_probs_per_token.mean()

# Cross-entropy via F.cross_entropy for comparison.
# F.cross_entropy expects shape [N, C] for logits and [N] for targets.
# We flatten (batch x posn-1) into N and use the sliced logits/tokens.
lp_flat = logits2[:, :-1].reshape(-1, D_VOCAB)  # [(batch*(posn-1)), d_vocab]
tgt_flat = tokens2[:, 1:].reshape(-1)  # [(batch*(posn-1))]
loss_ce = F.cross_entropy(lp_flat, tgt_flat)

print(f"manual loss   : {loss_manual.item():.6f}")
print(f"F.cross_entropy: {loss_ce.item():.6f}")
assert torch.isclose(loss_manual, loss_ce, atol=1e-5), "losses should match"
print("Losses match ✓")

# %%
