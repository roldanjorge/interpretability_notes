"""Comprehensive test suite for GPT-2 transformer modules.

47 tests across 9 test classes covering every module in the
section2 directory.
"""

import dataclasses

import einops
import torch as t
import torch.nn as nn
from transformer_lens.utils import gelu_new

from src.arena3.ch1_1.section2.attention import Attention
from src.arena3.ch1_1.section2.config import Config
from src.arena3.ch1_1.section2.demo_transformer import DemoTransformer
from src.arena3.ch1_1.section2.embed import Embed
from src.arena3.ch1_1.section2.layer_norm import LayerNorm
from src.arena3.ch1_1.section2.mlp import MLP
from src.arena3.ch1_1.section2.pos_embed import PosEmbed
from src.arena3.ch1_1.section2.transformer_block import TransformerBlock
from src.arena3.ch1_1.section2.unembed import Unembed


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TestConfig:
    def test_default_values(self):
        cfg = Config()
        assert cfg.d_model == 768
        assert cfg.debug is True
        assert cfg.layer_norm_eps == 1e-5
        assert cfg.d_vocab == 50257
        assert cfg.init_range == 0.02
        assert cfg.n_ctx == 1024
        assert cfg.d_head == 64
        assert cfg.d_mlp == 3072
        assert cfg.n_heads == 12
        assert cfg.n_layers == 12

    def test_custom_values(self):
        cfg = Config(d_model=32, d_vocab=128, n_ctx=16)
        assert cfg.d_model == 32
        assert cfg.d_vocab == 128
        assert cfg.n_ctx == 16

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(Config)
        cfg = Config()
        assert dataclasses.is_dataclass(cfg)


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------
class TestEmbed:
    def test_param_shapes(self, small_cfg):
        embed = Embed(small_cfg)
        assert embed.W_E.shape == (small_cfg.d_vocab, small_cfg.d_model)

    def test_weight_distribution(self, small_cfg):
        embed = Embed(small_cfg)
        assert abs(embed.W_E.std().item() - small_cfg.init_range) < 0.01

    def test_output_shape(self, small_cfg, device):
        embed = Embed(small_cfg).to(device)
        tokens = t.randint(0, small_cfg.d_vocab, (2, 8), device=device)
        out = embed(tokens)
        assert out.shape == (2, 8, small_cfg.d_model)

    def test_is_lookup(self, small_cfg, device):
        """Output should be equivalent to W_E[tokens]."""
        embed = Embed(small_cfg).to(device)
        tokens = t.randint(0, small_cfg.d_vocab, (2, 8), device=device)
        out = embed(tokens)
        expected = embed.W_E[tokens]
        assert t.allclose(out, expected, atol=1e-5)

    def test_single_token(self, small_cfg, device):
        embed = Embed(small_cfg).to(device)
        tokens = t.tensor([[5]], device=device)
        out = embed(tokens)
        assert out.shape == (1, 1, small_cfg.d_model)
        assert t.allclose(out[0, 0], embed.W_E[5], atol=1e-5)


# ---------------------------------------------------------------------------
# PosEmbed
# ---------------------------------------------------------------------------
class TestPosEmbed:
    def test_param_shape(self, small_cfg):
        pos_embed = PosEmbed(small_cfg)
        assert pos_embed.W_pos.shape == (small_cfg.n_ctx, small_cfg.d_model)

    def test_weight_distribution(self, small_cfg):
        pos_embed = PosEmbed(small_cfg)
        assert abs(pos_embed.W_pos.std().item() - small_cfg.init_range) < 0.01

    def test_output_shape(self, small_cfg, device):
        pos_embed = PosEmbed(small_cfg).to(device)
        tokens = t.randint(0, small_cfg.d_vocab, (2, 8), device=device)
        out = pos_embed(tokens)
        assert out.shape == (2, 8, small_cfg.d_model)

    def test_same_across_batch(self, small_cfg, device):
        """All batch elements should get the same positional embeddings."""
        pos_embed = PosEmbed(small_cfg).to(device)
        tokens = t.randint(0, small_cfg.d_vocab, (3, 8), device=device)
        out = pos_embed(tokens)
        assert t.allclose(out[0], out[1])
        assert t.allclose(out[0], out[2])

    def test_slices_w_pos_correctly(self, small_cfg, device):
        """Output should be W_pos[:seq_len] repeated across batch."""
        pos_embed = PosEmbed(small_cfg).to(device)
        seq_len = 8
        tokens = t.randint(0, small_cfg.d_vocab, (2, seq_len), device=device)
        out = pos_embed(tokens)
        expected = pos_embed.W_pos[:seq_len]
        assert t.allclose(out[0], expected, atol=1e-5)
        assert t.allclose(out[1], expected, atol=1e-5)


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------
class TestLayerNorm:
    def test_param_shapes(self, small_cfg):
        ln = LayerNorm(small_cfg)
        assert ln.w.shape == (small_cfg.d_model,)
        assert ln.b.shape == (small_cfg.d_model,)

    def test_param_values(self, small_cfg):
        """Weights should be ones, biases zeros."""
        ln = LayerNorm(small_cfg)
        assert t.allclose(ln.w, t.ones(small_cfg.d_model))
        assert t.allclose(ln.b, t.zeros(small_cfg.d_model))

    def test_output_shape(self, small_cfg, device):
        ln = LayerNorm(small_cfg).to(device)
        x = t.randn(2, 8, small_cfg.d_model, device=device)
        out = ln(x)
        assert out.shape == (2, 8, small_cfg.d_model)

    def test_zero_mean_unit_var(self, small_cfg, device):
        """With default w=1, b=0, output should have ~0 mean and ~1 var."""
        ln = LayerNorm(small_cfg).to(device)
        x = t.randn(2, 8, small_cfg.d_model, device=device) * 5 + 3
        out = ln(x)
        mean = out.mean(dim=-1)
        var = out.var(dim=-1, unbiased=False)
        assert t.allclose(mean, t.zeros_like(mean), atol=1e-5)
        assert t.allclose(var, t.ones_like(var), atol=1e-4)

    def test_learned_params(self, small_cfg, device):
        """Custom w and b should scale and shift the output."""
        ln = LayerNorm(small_cfg).to(device)
        ln.w.data = t.full((small_cfg.d_model,), 2.0, device=device)
        ln.b.data = t.full((small_cfg.d_model,), 1.0, device=device)
        x = t.randn(2, 8, small_cfg.d_model, device=device)
        out = ln(x)
        # Mean should be ~1 (the bias), var should be ~4 (scale^2)
        mean = out.mean(dim=-1)
        var = out.var(dim=-1, unbiased=False)
        assert t.allclose(mean, t.ones_like(mean), atol=1e-4)
        assert t.allclose(var, t.full_like(var, 4.0), atol=0.1)

    def test_matches_nn_layer_norm(self, small_cfg, device):
        """Should produce the same result as nn.LayerNorm."""
        ln = LayerNorm(small_cfg).to(device)
        ref = nn.LayerNorm(small_cfg.d_model, eps=small_cfg.layer_norm_eps).to(device)
        ref.weight.data = ln.w.data.clone()
        ref.bias.data = ln.b.data.clone()
        x = t.randn(2, 8, small_cfg.d_model, device=device)
        assert t.allclose(ln(x), ref(x), atol=1e-5)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
class TestAttention:
    def test_param_shapes(self, small_cfg):
        attn = Attention(small_cfg)
        assert attn.W_Q.shape == (small_cfg.n_heads, small_cfg.d_model, small_cfg.d_head)
        assert attn.W_K.shape == (small_cfg.n_heads, small_cfg.d_model, small_cfg.d_head)
        assert attn.W_V.shape == (small_cfg.n_heads, small_cfg.d_model, small_cfg.d_head)
        assert attn.W_O.shape == (small_cfg.n_heads, small_cfg.d_head, small_cfg.d_model)
        assert attn.b_Q.shape == (small_cfg.n_heads, small_cfg.d_head)
        assert attn.b_K.shape == (small_cfg.n_heads, small_cfg.d_head)
        assert attn.b_V.shape == (small_cfg.n_heads, small_cfg.d_head)
        assert attn.b_O.shape == (small_cfg.d_model,)

    def test_bias_zeros(self, small_cfg):
        attn = Attention(small_cfg)
        assert t.allclose(attn.b_Q, t.zeros_like(attn.b_Q))
        assert t.allclose(attn.b_K, t.zeros_like(attn.b_K))
        assert t.allclose(attn.b_V, t.zeros_like(attn.b_V))
        assert t.allclose(attn.b_O, t.zeros_like(attn.b_O))

    def test_ignore_buffer(self, small_cfg):
        attn = Attention(small_cfg)
        assert hasattr(attn, "IGNORE")
        assert attn.IGNORE.item() == float("-inf")

    def test_output_shape(self, small_cfg, device):
        attn = Attention(small_cfg).to(device)
        x = t.randn(2, 8, small_cfg.d_model, device=device)
        out = attn(x)
        assert out.shape == (2, 8, small_cfg.d_model)

    def test_deterministic(self, small_cfg, device):
        attn = Attention(small_cfg).to(device)
        x = t.randn(2, 8, small_cfg.d_model, device=device)
        out1 = attn(x)
        out2 = attn(x)
        assert t.allclose(out1, out2)

    def test_causal_mask_upper_triangle(self, small_cfg, device):
        """Upper triangle of attention scores should be -inf after masking."""
        attn = Attention(small_cfg).to(device)
        seq_len = 8
        scores = t.zeros(1, small_cfg.n_heads, seq_len, seq_len, device=device)
        masked = attn.apply_causal_mask(scores)
        upper = t.triu(t.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        assert (masked[0, 0][upper] == float("-inf")).all()

    def test_mask_preserves_lower_triangle(self, small_cfg, device):
        """Lower triangle and diagonal should be preserved after masking."""
        attn = Attention(small_cfg).to(device)
        seq_len = 8
        scores = t.randn(1, small_cfg.n_heads, seq_len, seq_len, device=device)
        original = scores.clone()
        masked = attn.apply_causal_mask(scores)
        lower = t.tril(t.ones(seq_len, seq_len, device=device)).bool()
        assert t.allclose(masked[0, 0][lower], original[0, 0][lower])

    def test_causal_property(self, small_cfg, device):
        """Changing future tokens shouldn't affect past token outputs."""
        attn = Attention(small_cfg).to(device)
        x1 = t.randn(1, small_cfg.n_ctx, small_cfg.d_model, device=device)
        x2 = x1.clone()
        x2[:, -1, :] = t.randn(small_cfg.d_model, device=device)
        out1 = attn(x1)
        out2 = attn(x2)
        assert t.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------
class TestMLP:
    def test_param_shapes(self, small_cfg):
        mlp_mod = MLP(small_cfg)
        assert mlp_mod.W_in.shape == (small_cfg.d_model, small_cfg.d_mlp)
        assert mlp_mod.W_out.shape == (small_cfg.d_mlp, small_cfg.d_model)
        assert mlp_mod.b_in.shape == (small_cfg.d_mlp,)
        assert mlp_mod.b_out.shape == (small_cfg.d_model,)

    def test_bias_zeros(self, small_cfg):
        mlp_mod = MLP(small_cfg)
        assert t.allclose(mlp_mod.b_in, t.zeros_like(mlp_mod.b_in))
        assert t.allclose(mlp_mod.b_out, t.zeros_like(mlp_mod.b_out))

    def test_output_shape(self, small_cfg, device):
        mlp_mod = MLP(small_cfg).to(device)
        x = t.randn(2, 8, small_cfg.d_model, device=device)
        out = mlp_mod(x)
        assert out.shape == (2, 8, small_cfg.d_model)

    def test_gelu_nonlinearity(self, small_cfg, device):
        """Output should differ from a purely linear transformation."""
        mlp_mod = MLP(small_cfg).to(device)
        x = t.randn(2, 8, small_cfg.d_model, device=device)
        out = mlp_mod(x)
        # Linear-only path (skip the gelu)
        linear_out = (
            einops.einsum(
                einops.einsum(x, mlp_mod.W_in, "b p d, d m -> b p m") + mlp_mod.b_in,
                mlp_mod.W_out,
                "b p m, m d -> b p d",
            )
            + mlp_mod.b_out
        )
        assert not t.allclose(out, linear_out, atol=1e-3)

    def test_deterministic(self, small_cfg, device):
        mlp_mod = MLP(small_cfg).to(device)
        x = t.randn(2, 8, small_cfg.d_model, device=device)
        out1 = mlp_mod(x)
        out2 = mlp_mod(x)
        assert t.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Unembed
# ---------------------------------------------------------------------------
class TestUnembed:
    def test_param_shapes(self, small_cfg):
        unembed = Unembed(small_cfg)
        assert unembed.W_U.shape == (small_cfg.d_model, small_cfg.d_vocab)
        assert unembed.b_U.shape == (small_cfg.d_vocab,)

    def test_bias_zero_initialized(self, small_cfg):
        """b_U should be zero-initialized."""
        unembed = Unembed(small_cfg)
        assert t.allclose(unembed.b_U, t.zeros(small_cfg.d_vocab))

    def test_weight_distribution(self, small_cfg):
        unembed = Unembed(small_cfg)
        assert abs(unembed.W_U.std().item() - small_cfg.init_range) < 0.01

    def test_output_shape(self, small_cfg, device):
        unembed = Unembed(small_cfg).to(device)
        x = t.randn(2, 8, small_cfg.d_model, device=device)
        out = unembed(x)
        assert out.shape == (2, 8, small_cfg.d_vocab)

    def test_numerical_correctness(self, small_cfg, device):
        """Forward output should match manual einsum + bias."""
        unembed = Unembed(small_cfg).to(device)
        x = t.randn(2, 8, small_cfg.d_model, device=device)
        out = unembed(x)
        expected = einops.einsum(x, unembed.W_U, "b p d, d v -> b p v") + unembed.b_U
        assert t.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------
class TestTransformerBlock:
    def test_submodules_exist(self, small_cfg):
        block = TransformerBlock(small_cfg)
        assert hasattr(block, "ln1")
        assert hasattr(block, "attn")
        assert hasattr(block, "ln2")
        assert hasattr(block, "mlp")

    def test_submodule_types(self, small_cfg):
        block = TransformerBlock(small_cfg)
        assert isinstance(block.ln1, LayerNorm)
        assert isinstance(block.attn, Attention)
        assert isinstance(block.ln2, LayerNorm)
        assert isinstance(block.mlp, MLP)

    def test_output_shape(self, small_cfg, device):
        block = TransformerBlock(small_cfg).to(device)
        x = t.randn(2, 8, small_cfg.d_model, device=device)
        out = block(x)
        assert out.shape == (2, 8, small_cfg.d_model)

    def test_residual_connections(self, small_cfg, device):
        """Manual pipeline should match forward output."""
        block = TransformerBlock(small_cfg).to(device)
        x = t.randn(2, 8, small_cfg.d_model, device=device)
        resid_mid = block.attn(block.ln1(x)) + x
        resid_post = block.mlp(block.ln2(resid_mid)) + resid_mid
        out = block(x)
        assert t.allclose(out, resid_post, atol=1e-5)

    def test_deterministic(self, small_cfg, device):
        block = TransformerBlock(small_cfg).to(device)
        x = t.randn(2, 8, small_cfg.d_model, device=device)
        out1 = block(x)
        out2 = block(x)
        assert t.allclose(out1, out2)


# ---------------------------------------------------------------------------
# DemoTransformer
# ---------------------------------------------------------------------------
class TestDemoTransformer:
    def test_submodules_exist(self, small_cfg):
        model = DemoTransformer(small_cfg)
        assert hasattr(model, "embed")
        assert hasattr(model, "pos_embed")
        assert hasattr(model, "blocks")
        assert hasattr(model, "ln_final")
        assert hasattr(model, "unembed")

    def test_block_count(self, small_cfg):
        model = DemoTransformer(small_cfg)
        assert len(model.blocks) == small_cfg.n_layers

    def test_output_shape(self, small_cfg, device):
        model = DemoTransformer(small_cfg).to(device)
        tokens = t.randint(0, small_cfg.d_vocab, (2, 8), device=device)
        out = model(tokens)
        assert out.shape == (2, 8, small_cfg.d_vocab)

    def test_matches_manual_pipeline(self, small_cfg, device):
        """Manual pipeline should match forward output."""
        model = DemoTransformer(small_cfg).to(device)
        tokens = t.randint(0, small_cfg.d_vocab, (2, 8), device=device)
        x = model.embed(tokens) + model.pos_embed(tokens)
        for block in model.blocks:
            x = block(x)
        x = model.ln_final(x)
        expected = model.unembed(x)
        out = model(tokens)
        assert t.allclose(out, expected, atol=1e-5)

    def test_produces_valid_logits(self, small_cfg, device):
        """Softmax of output should sum to ~1."""
        model = DemoTransformer(small_cfg).to(device)
        tokens = t.randint(0, small_cfg.d_vocab, (2, 8), device=device)
        out = model(tokens)
        probs = out.softmax(dim=-1)
        sums = probs.sum(dim=-1)
        assert t.allclose(sums, t.ones_like(sums), atol=1e-5)
