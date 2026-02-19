import torch as t
import torch.nn.functional as F
import torch.nn as nn

from src.arena3.ch0_2.section1.rely import ReLU
from src.arena3.ch0_2.section1.linear import Linear
from src.arena3.ch0_2.section1.flatten import Flatten
from src.arena3.ch0_2.section1.simple_mlp import SimpleMLP


def compare_module_attributes(custom_module, reference_module):
    module_name = custom_module.__class__.__name__

    # Compare named parameters
    custom_params = dict(custom_module.named_parameters())
    ref_params = dict(reference_module.named_parameters())

    assert list(custom_params.keys()) == list(ref_params.keys()), (
        f"Your {module_name} should declare the following parameters in order.\n"
        f"Expected: {list(ref_params.keys())}\nActual: {list(custom_params.keys())}"
    )

    # Check tensor shapes for parameters
    for name in custom_params:
        if hasattr(custom_params[name], "shape") and hasattr(ref_params[name], "shape"):
            assert custom_params[name].shape == ref_params[name].shape, (
                f"Shape mismatch for parameter '{name}' in {module_name}.\n"
                f"Expected shape: {ref_params[name].shape}, Actual shape: {custom_params[name].shape}"
            )

    # Compare named buffers
    custom_buffers = dict(custom_module.named_buffers())
    ref_buffers = dict(reference_module.named_buffers())

    assert list(custom_buffers.keys()) == list(ref_buffers.keys()), (
        f"Your {module_name} should declare the following buffers in order.\n"
        f"Expected: {list(ref_buffers.keys())}\nActual: {list(custom_buffers.keys())}"
    )

    # Check tensor shapes for buffers
    for name in custom_buffers:
        if hasattr(custom_buffers[name], "shape") and hasattr(ref_buffers[name], "shape"):
            assert custom_buffers[name].shape == ref_buffers[name].shape, (
                f"Shape mismatch for buffer '{name}' in {module_name}.\n"
                f"Expected shape: {ref_buffers[name].shape}, Actual shape: {custom_buffers[name].shape}"
            )

# --------------------------------------------------------------
# ReLU
# --------------------------------------------------------------
class TestReLU:
    def test_relu(self):
        x = t.randn(10) - 0.5
        actual = ReLU()(x)
        expected = F.relu(x)
        t.testing.assert_close(actual, expected)
        print("All tests in `test_relu` passed!")


# --------------------------------------------------------------
# Linear
# --------------------------------------------------------------
class TestLinear:
    def test_linear_forward(self, bias=False):
        """Your Linear should produce identical results to torch.nn given identical parameters."""
        x = t.rand((10, 512))
        yours = Linear(512, 64, bias=bias)
        official = t.nn.Linear(512, 64, bias=bias)
        # ensure the weights are the same
        yours.load_state_dict(official.state_dict())
        actual = yours(x)
        expected = official(x)
        t.testing.assert_close(actual, expected)
        print("All tests in `test_linear_forward` passed!")


    def test_linear_parameters(self, bias=False):
        l = Linear(2, 3, bias=bias)
        l_sol = nn.Linear(2, 3, bias=bias)
        compare_module_attributes(l, l_sol)
        if not bias:
            assert l.bias is None, "Bias should be None when not enabled."
        print("All tests in `test_linear_parameters` passed!")

    def test_linear_forward_with_bias(self):
        """Linear with bias=True should match torch.nn.Linear."""
        x = t.rand((10, 512))
        yours = Linear(512, 64, bias=True)
        official = t.nn.Linear(512, 64, bias=True)
        yours.load_state_dict(official.state_dict())
        actual = yours(x)
        expected = official(x)
        t.testing.assert_close(actual, expected)

    def test_linear_parameters_with_bias(self):
        l = Linear(2, 3, bias=True)
        l_sol = nn.Linear(2, 3, bias=True)
        compare_module_attributes(l, l_sol)
        assert l.bias is not None, "Bias should be a Parameter when enabled."

    def test_extra_repr(self):
        l = Linear(4, 8, bias=True)
        assert "in_features=4" in l.extra_repr()
        assert "out_features=8" in l.extra_repr()
        assert "bias=True" in l.extra_repr()

        l_no_bias = Linear(4, 8, bias=False)
        assert "bias=False" in l_no_bias.extra_repr()


# --------------------------------------------------------------
# Flatten
# --------------------------------------------------------------
class TestFlatten:
    def test_default_dims(self):
        """Default (start_dim=1, end_dim=-1) should match t.flatten."""
        x = t.randn(2, 3, 4, 5)
        actual = Flatten()(x)
        expected = t.flatten(x, 1, -1)
        t.testing.assert_close(actual, expected)
        assert actual.shape == (2, 60)

    def test_custom_start_and_end(self):
        """Flatten middle dims only."""
        x = t.randn(2, 3, 4, 5)
        actual = Flatten(start_dim=1, end_dim=2)(x)
        expected = t.flatten(x, 1, 2)
        t.testing.assert_close(actual, expected)
        assert actual.shape == (2, 12, 5)

    def test_negative_end_dim(self):
        """Negative end_dim should resolve correctly."""
        x = t.randn(2, 3, 4, 5)
        actual = Flatten(start_dim=0, end_dim=-2)(x)
        expected = t.flatten(x, 0, -2)
        t.testing.assert_close(actual, expected)
        assert actual.shape == (24, 5)

    def test_nonnegative_end_dim(self):
        """Positive end_dim should work without negative index conversion."""
        x = t.randn(2, 3, 4, 5)
        actual = Flatten(start_dim=1, end_dim=3)(x)
        expected = t.flatten(x, 1, 3)
        t.testing.assert_close(actual, expected)
        assert actual.shape == (2, 60)

    def test_single_dim_noop(self):
        """Flattening a single dim should be a no-op."""
        x = t.randn(2, 3, 4)
        actual = Flatten(start_dim=1, end_dim=1)(x)
        t.testing.assert_close(actual, x)
        assert actual.shape == x.shape

    def test_flatten_all_dims(self):
        """Flattening all dims should produce a 1-D tensor."""
        x = t.randn(2, 3, 4)
        actual = Flatten(start_dim=0, end_dim=-1)(x)
        assert actual.shape == (24,)

    def test_extra_repr(self):
        f = Flatten(start_dim=2, end_dim=-1)
        r = f.extra_repr()
        assert "start_dim=2" in r
        assert "end_dim=-1" in r


# --------------------------------------------------------------
# SimpMLP 
# --------------------------------------------------------------
class TestSimpleMLP:
    def test_mlp_module(self):
        import tests.arena3.ch0_2.section1.solutions as solutions

        mlp: nn.Module = SimpleMLP()
        num_params = sum(p.numel() for p in mlp.parameters())
        assert num_params == 79510, (
            f"Expected (28*28 + 1) * 100 + ((100 + 1) * 10) = 79510 parameters, got {num_params}"
        )
        mlp_sol = solutions.SimpleMLP()
        compare_module_attributes(mlp, mlp_sol)
        print("All tests in `test_mlp_module` passed!")


    def test_mlp_forward(self):
        import tests.arena3.ch0_2.section1.solutions as solutions

        mlp: nn.Module = SimpleMLP()
        mlp_sol = solutions.SimpleMLP()
        x = t.rand((10, 28, 28))

        # ensure the weights are the same
        mlp.load_state_dict(mlp_sol.state_dict())

        out = mlp(x)
        out_sol = mlp_sol(x)
        t.testing.assert_close(out, out_sol)
        print("All tests in `test_mlp_forward` passed!")