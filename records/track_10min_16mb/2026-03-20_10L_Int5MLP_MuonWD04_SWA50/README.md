# 10L Int5-MLP + MuonWD=0.04 + SWA/50 + SmearGate + BigramHash

**val_bpb: 1.14526** (sliding window stride=64, post int6+zstd quantization roundtrip)

## Key Innovation: Mixed Int5/Int6 Quantization

Instead of uniform int6 quantization for all weights, use:
- **Int5 [-16,15]** for MLP weights (largest tensors, most compressible)
- **Int6 [-32,31]** for attention weights (more precision-sensitive)
- **FP16** for tied embeddings and last-layer key projections

Int5 values stored in int8 bytes have **3 zero high bits** vs 2 for int6. zstd-22 compresses int5 at 1.88x vs int6 at 1.51x — saving **1.86MB** (14.07MB vs 16.05MB). This funds a **10th transformer layer** while staying under 16MB.

## Architecture
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu² activation
- SmearGate + BigramHash (4096 buckets, dim=128)
- Orthogonal init with muP-scaled output projections
- U-Net skip connections

## Training
- Weight decay: 0.04 (Muon + AdamW) — improves quantization friendliness
- SWA: every 50 steps, ~29 checkpoint average during warmdown
- Muon optimizer: momentum 0.99, warmup 0.92→0.99 over 1500 steps
- matrix_lr=0.02, scalar_lr=0.02, tied_embed_lr=0.03
- seq_len=2048, batch=786K tokens, warmdown=3000 iters
- Sliding window eval: stride=64

## Metrics
- Model params: 24,730,705
- Steps: 6,694 in 600s (89.5 ms/step)
- Pre-quant val_bpb: 1.1748
- Post-quant (int6+zstd roundtrip): **val_bpb: 1.14526**
- Artifact: 15,521,020 bytes (15.52MB)

## Run Command
```bash
NUM_LAYERS=10 bash eval/eval.sh
```
All other defaults are baked into train_gpt.py.

## Ablation Summary
| Change | BPB | Delta |
|--------|-----|-------|
| 9L int6 (PR162 base) | 1.14847 | baseline |
| + int5 MLP (9L) | 1.15663 | +0.008 (quant cost) |
| + 10th layer (int5 MLP) | 1.14803 | -0.0005 (depth compensates) |
| + WD=0.04 + SWA/50 | **1.14526** | **-0.003** |

Built on PR #162 by @unnir (SmearGate, BigramHash, OrthoInit).
