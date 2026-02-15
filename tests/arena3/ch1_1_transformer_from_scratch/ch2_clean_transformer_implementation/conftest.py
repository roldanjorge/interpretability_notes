"""Pytest configuration for transformer module tests."""

import pytest
import torch as t

from src.arena3.ch1_1_transformer_from_scratch.ch2_clean_transformer_implementation.config import Config


@pytest.fixture
def small_cfg():
    """A small Config for fast tests (d_model=32, 4 heads, 2 layers)."""
    return Config(
        d_model=32,
        d_vocab=128,
        n_ctx=16,
        d_head=8,
        d_mlp=64,
        n_heads=4,
        n_layers=2,
    )


@pytest.fixture
def device():
    """CPU device for deterministic testing."""
    return t.device("cpu")


@pytest.fixture(autouse=True)
def seed():
    """Set a fixed random seed before every test."""
    t.manual_seed(42)
