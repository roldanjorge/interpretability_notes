# Interpretability Notes

Personal study notes and from-scratch implementations following the [ARENA 3.0](https://www.arena.education/) mechanistic interpretability curriculum.

## Structure

```
src/arena3/
  ch1_1/
    section2/   # Clean transformer implementation (GPT-2 style)
    section3/   # Training a transformer + sampling
  utils/        # Shared utilities (device selection, log probs)
docs/           # LaTeX write-ups and PDFs
tests/          # Test helpers comparing against TransformerLens
```

## Setup

```bash
uv sync          # install dependencies
uv run python <file>
```

Files use `# %%` cell markers for cell-by-cell execution in VS Code.
