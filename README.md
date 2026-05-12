# tausurv & causurv

Monorepo containing:

- **tausurv** — survival analysis
- **causurv** — causal survival analysis (rename to `dosurv` pending team decision)

## Setup

PyTorch must be installed with the build matching your hardware (CPU, CUDA, ROCm). Install it first, then sync the workspace.

```bash
# Pick one:
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install torch --index-url https://download.pytorch.org/whl/cu130

# Then sync everything else:
uv sync
```

## License

MIT
