# Multimodal Finetuned Representations

Geometric interpretability study to analyze what changes and where inside the LLM backbone of vision-language models after visual instruction finetuning.

## Setup

<details>
<summary><b>Click to expand setup instructions</b></summary>

### Requirements

**Critical constraint:** Python 3.12 is required - Flash Attention 2.8.3 only provides pre-built wheels for this version.

**Compatible hardware:**
- NVIDIA A100 (CUDA 12.2) ✓
- NVIDIA H100 (CUDA 13.0) ✓
- Any NVIDIA GPU with CUDA runtime ≥ 12.1

**Dependency chain:**
```
Python 3.12 → Flash Attention 2.8.3 (pre-built wheels, 7-19% faster attention)
           → PyTorch 2.4.1+cu121 (forward-compatible with CUDA 12.1+)
           → Works on both A100 and H100 without modification
```

**Tools:**
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
uv sync
source .venv/bin/activate
```

No CUDA toolkit installation needed - all dependencies use pre-built wheels.

The `uv.lock` file ensures exact environment replication. However, different hardware may require adjustments (especially for Flash Attention). If you encounter installation issues:

1. Remove `flash-attn` from `pyproject.toml
2. Run `uv sync`
3. Manually install: `uv pip install flash-attn==2.8.3 --no-build-isolation`

</details>

## A note of caution

The `.env` shared file is currently available with settings for usage within ORFEO.

> ⚠️⚠️  If the repository is made public, make sure to adjust the `.env` file accordingly to avoid exposing sensitive information. ⚠️⚠️
