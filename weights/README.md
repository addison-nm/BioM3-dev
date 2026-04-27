# Weights directory

This directory is gitignored. It contains pre-trained model weights for the BioM3 pipeline:

| Subdirectory | Contents |
|---|---|
| `LLMs/` | Backbone language models (ESM-2, BiomedBERT) used to compile PenCL |
| `PenCL/` | Stage 1 — PenCL encoder weights (`.bin` or `.ckpt`) |
| `Facilitator/` | Stage 2 — Facilitator alignment weights (`.bin`) |
| `ProteoScribe/` | Stage 3 — ProteoScribe diffusion model weights (`.bin` or `.ckpt`) |

## Setup

Weights are synced from a shared directory on each machine using the sync script:

```bash
./scripts/link_weights.sh <shared_weights_path> weights
```

See [docs/setup_shared_weights.md](../docs/setup_shared_weights.md) for machine-specific shared paths, the full list of required files, and setup instructions.
