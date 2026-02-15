# Transformer Module Tests

## Setup

Install dev dependencies:

```bash
uv sync --group dev
```

## Running Tests

Run all tests:

```bash
uv run pytest tests/arena3/1_1_transformer_from_scratch/2_clean_transformer_implementation/ -v
```

Run with coverage:

```bash
uv run pytest tests/arena3/1_1_transformer_from_scratch/2_clean_transformer_implementation/ \
  --cov=src/arena3/1_1_transformer_from_scratch/2_clean_transformer_implementation \
  --cov-report=term-missing -v
```

Run a single test class:

```bash
uv run pytest tests/arena3/1_1_transformer_from_scratch/2_clean_transformer_implementation/test_transformer_modules.py::TestAttention -v
```

Run a single test:

```bash
uv run pytest tests/arena3/1_1_transformer_from_scratch/2_clean_transformer_implementation/test_transformer_modules.py::TestAttention::test_causal_property -v
```
