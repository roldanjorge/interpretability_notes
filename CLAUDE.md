# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal study notes and from-scratch implementations following the [ARENA 3.0](https://www.arena.education/) mechanistic interpretability curriculum. Uses `uv` for package management with a `pyproject.toml`-based setup.

## Build & Run

```bash
uv sync                  # install dependencies
uv run python <file>     # run a script
```

Python files use `# %%` cell markers for interactive execution in VS Code (Jupyter-style). Most files are meant to be run cell-by-cell, not as standalone scripts.

## Project Structure

```
src/
  arena3/
    utils/device.py         # shared device selection (mps > cuda > cpu)
    <chapter>/<section>/    # each ARENA section is a subdirectory
tests/
  arena3/                   # test utilities (rand_float_test, rand_int_test, load_gpt2_test)
```

Chapters follow the ARENA numbering: `ch1_1/` (transformer from scratch), with subsections like `section1/` (inputs & outputs) and `section2/` (clean implementation).

## Architecture Notes

- **Transformer components** are split into individual modules: `config.py`, `embed.py`, `pos_embed.py`, `attention.py`, `mlp.py`, `layer_norm.py`, `unembed.py`, `transformer_block.py`, `demo_transformer.py`. Each module defines a single `nn.Module` subclass.
- Components use `einops.einsum` for tensor operations and `jaxtyping` annotations on forward signatures (e.g., `Float[Tensor, "batch posn d_model"]`).
- `Config` is a `@dataclass` with GPT-2 small defaults (d_model=768, n_heads=12, n_layers=12, d_vocab=50257).
- Implementations include both "JR Solution" (personal) and commented-out "Reference solution" variants.
- Testing uses `load_gpt2_test` which loads real GPT-2 weights into custom modules and compares outputs against TransformerLens's `HookedTransformer`.

## Conventions

- `torch` is imported as `t` (not `torch`)
- Device is resolved via `src.arena3.utils.device` or inline
- All imports use absolute paths (e.g., `from src.arena3.ch1_1.section2.config import Config`)
