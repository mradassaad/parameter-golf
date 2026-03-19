# 10L Mixed-Precision: Lower LR + FP16 Tied Embedding + Int6 Middle Layers

## Summary

10-layer transformer with lower learning rates, fp16 tied embedding export, and int6 quantization for middle layers (2-7). This achieves **val_bpb = 1.2129** (mean across 5 seeds), a **0.0248 nat improvement** over the naive baseline.

### Key Changes from Baseline

1. **10 transformer layers** (vs 9 baseline) — more effective depth for the same architecture
2. **Lower learning rates**: MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03 — reduces post-quantization degradation and improves compression
3. **FP16 tied embedding export** — keeps `tok_emb.weight` in fp16 instead of int8 during export, nearly eliminating the quantization gap on the most critical tensor (shared between embedding and output head)
4. **Int6 middle layers** (layers 2-7) — quantizes middle layer weights to 64 effective levels (multiples of 4 in int8) instead of 256, saving ~2MB to fit under 16MB budget

### Architecture

- 10 transformer blocks, 512 model dim, 8 attention heads, 4 KV heads
- GQA attention with RoPE, ReLU² MLP (2x expansion)
- Tied embeddings with 1024 BPE vocabulary
- U-Net style skip connections (5 encoder + 5 decoder layers)

### Export Strategy

The 10-layer model has 18.9M parameters, which exceeds 16MB with standard int8+zlib. Our mixed-precision export strategy:

| Layer Group | Precision | Rationale |
|---|---|---|
| tok_emb.weight (tied) | fp16 (1024 levels) | Most sensitive — serves as both embedding and output head |
| Layers 0-1 (early) | int8 (256 levels) | Critical for input processing |
| Layers 2-7 (middle) | int6 (64 levels) | Less sensitive, saves ~2MB |
| Layers 8-9 (late) | int8 (256 levels) | Critical for output quality |

## Multi-Seed Results (5 seeds, p << 0.001)

| Seed | val_loss (nats) | val_bpb | Artifact |
|---|---|---|---|
| 1337 | 2.04683482 | 1.21225087 | 15,361,072 |
| 42 | 2.04998529 | 1.21411676 | 15,358,320 |
| 123 | 2.04799422 | 1.21293753 | 15,355,036 |
| 7 | 2.04579765 | 1.21163660 | 15,362,933 |
| 2024 | 2.04872971 | 1.21337313 | 15,362,170 |
| **Mean** | **2.04786834** | **1.21286298** | **15,359,906** |
| **Std** | **0.00162751** | | |

- **Mean improvement: 0.0248 nats** (5.0× the 0.005 threshold)
- **t-statistic: 34.12** (df=4, one-sided t critical for p<0.01 = 3.747)
- **p << 0.001**
- All 5 runs individually beat SOTA by >0.005 nats
- All 5 artifacts under 16MB

## Hardware

All runs on 8×H100 SXM (RunPod). Each run completes in ~600s (wallclock cap), reaching ~12,350 training steps at ~48.6ms/step.

## How to Run

```bash
RUN_ID=submission \
SEED=1337 \
NUM_LAYERS=10 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
FP16_EMBED_EXPORT=1 \
INT6_LAYER_START=2 \
INT6_LAYER_END=7 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
