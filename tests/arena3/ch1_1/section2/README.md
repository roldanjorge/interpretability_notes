# Transformer Module Tests

## Setup

Install dev dependencies:

```bash
uv sync --group dev
```

## Running Tests

Run all tests:

```bash
uv run pytest tests/arena3/ch1_1/section2/ -v
```

Run with coverage:

```bash
uv run pytest tests/arena3/ch1_1/section2/ \
  --cov=src/arena3/ch1_1/section2 \
  --cov-report=term-missing -v
```

Run a single test class:

```bash
uv run pytest tests/arena3/ch1_1/section2/test_transformer_modules.py::TestAttention -v
```

Run a single test:

```bash
uv run pytest tests/arena3/ch1_1/section2/test_transformer_modules.py::TestAttention::test_causal_property -v
```
