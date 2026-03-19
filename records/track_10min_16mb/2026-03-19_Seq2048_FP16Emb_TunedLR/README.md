# Seq2048 + FP16 Tied Embedding + Tuned LR

## Summary

Five changes stacked on the Naive Baseline, each targeting a known inefficiency:

1. **Sequence length 2048** (from 1024): Longer context per training example improves modeling quality despite fewer total steps (~10,400 vs ~12,100).

2. **FP16 tied embedding passthrough**: The tied embedding matrix serves dual duty as both input embedding and output projection head. INT8 quantization introduces ~0.007 BPB of noise. Keeping it in FP16 during export reduces the quant gap from ~0.007 to ~0.0003 BPB.

3. **MLP hidden dim 960** (from 1024): Trimmed to compensate for the extra bytes from the FP16 embedding while staying under the 16MB cap.

4. **Lower learning rates**: MATRIX_LR=0.032, SCALAR_LR=0.032, TIED_EMBED_LR=0.04 (from 0.04, 0.04, 0.05). Multiple independent experiments suggest slightly lower LRs improve final convergence at this model scale.

5. **Warmdown 3600** (from 1200): The default warmdown was calibrated for 20,000 steps but training is wallclock-capped at ~10,000 steps. The original warmdown never activated. Increasing to 3600 ensures proper LR decay in the final training phase.

## Configuration

```bash
RUN_ID=v1_seq2048_fp16emb \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=100 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_HIDDEN=960`
Tied embeddings with FP16 export passthrough.

## Results

| Seed | Steps | val_loss | val_bpb | Artifact size |
|------|-------|----------|---------|---------------|
| 1337 | 10,408 | 2.0370 | 1.2064 | 15,632,845 |
| 42 | 10,403 | 2.0383 | 1.2072 | 15,635,682 |
| 3 | 10,375 | 2.0370 | 1.2064 | 15,633,777 |

**Mean val_bpb: 1.2067** (std: 0.00044)
**Mean val_loss: 2.0374** (std: 0.00074)

Statistical significance vs SOTA (1.2244 BPB / 2.0727 val_loss):
- Improvement: 0.0353 nats (threshold: 0.005)
- t-statistic: -70.69, df=2, p << 0.01

Hardware: 8xH100 80GB HBM3, PyTorch 2.8.0+cu128, ~57.65ms/step avg.

## Included Files

- `train_gpt.py` (modified training script)
- `train_seed1337.log` (seed 1337 training log)
- `train_seed42.log` (seed 42 training log)
- `train_seed3.log` (seed 3 training log)
- `submission.json` (leaderboard metadata)
