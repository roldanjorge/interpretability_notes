"""Tests for get_log_probs utility."""

import torch as t

from src.arena3.utils.utils import get_log_probs


class TestGetLogProbs:
    def test_output_shape(self):
        """Output should be (batch, posn-1)."""
        batch, posn, d_vocab = 2, 10, 50
        logits = t.randn(batch, posn, d_vocab)
        tokens = t.randint(0, d_vocab, (batch, posn))
        out = get_log_probs(logits, tokens)
        assert out.shape == (batch, posn - 1)

    def test_single_position_pair(self):
        """With posn=2, output should have exactly 1 position."""
        batch, d_vocab = 3, 20
        logits = t.randn(batch, 2, d_vocab)
        tokens = t.randint(0, d_vocab, (batch, 2))
        out = get_log_probs(logits, tokens)
        assert out.shape == (batch, 1)

    def test_values_are_log_probabilities(self):
        """All output values should be <= 0 (log probs)."""
        logits = t.randn(2, 8, 30)
        tokens = t.randint(0, 30, (2, 8))
        out = get_log_probs(logits, tokens)
        assert (out <= 0).all()

    def test_manual_computation(self):
        """Output should match hand-computed log probs for a small example."""
        d_vocab = 4
        # batch=1, posn=3 -> output should be (1, 2)
        logits = t.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]])
        tokens = t.tensor([[0, 2, 1]])  # next-token targets are tokens[:, 1:] = [2, 1]

        out = get_log_probs(logits, tokens)

        log_probs = logits.log_softmax(dim=-1)
        # Position 0 predicts token 2, position 1 predicts token 1
        expected_0 = log_probs[0, 0, 2]  # logit at pos 0 for token 2
        expected_1 = log_probs[0, 1, 1]  # logit at pos 1 for token 1
        assert t.allclose(out, t.tensor([[expected_0, expected_1]]), atol=1e-5)

    def test_confident_prediction(self):
        """When logits strongly favor the correct next token, log prob should be near 0."""
        d_vocab = 5
        logits = t.zeros(1, 3, d_vocab)
        logits[0, 0, 2] = 100.0  # position 0 strongly predicts token 2
        tokens = t.tensor([[0, 2, 0]])  # next token at pos 0 is 2
        out = get_log_probs(logits, tokens)
        assert out[0, 0].item() > -0.01  # near 0

    def test_uniform_logits(self):
        """Uniform logits should give log(1/d_vocab) for every position."""
        d_vocab = 10
        logits = t.zeros(2, 5, d_vocab)  # uniform distribution
        tokens = t.randint(0, d_vocab, (2, 5))
        out = get_log_probs(logits, tokens)
        expected = t.full_like(out, -t.tensor(float(d_vocab)).log().item())
        assert t.allclose(out, expected, atol=1e-5)

    def test_batch_independence(self):
        """Each batch element should be computed independently."""
        d_vocab = 20
        logits = t.randn(3, 6, d_vocab)
        tokens = t.randint(0, d_vocab, (3, 6))
        full_out = get_log_probs(logits, tokens)
        for b in range(3):
            single_out = get_log_probs(logits[b : b + 1], tokens[b : b + 1])
            assert t.allclose(full_out[b], single_out[0], atol=1e-5)

    def test_different_vocab_sizes(self):
        """Should work with various vocabulary sizes."""
        for d_vocab in [2, 100, 1000]:
            logits = t.randn(1, 4, d_vocab)
            tokens = t.randint(0, d_vocab, (1, 4))
            out = get_log_probs(logits, tokens)
            assert out.shape == (1, 3)
