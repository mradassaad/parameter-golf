"""Profile training steps: baseline vs torch.compile.
Outputs chrome traces and summary tables."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
import time
import copy

# Set env vars to match our standard config
for k, v in {
    "FP16_INPROJ_ROWS": "0", "WARMDOWN_ITERS": "2600", "WARMDOWN_SHAPE": "linear",
    "MUON_EQ_R": "1", "LATE_QAT_THRESHOLD": "0.15", "WEIGHT_DECAY": "0.04",
    "MUON_MOMENTUM": "0.99", "MATRIX_LR": "0.025", "EVAL_STRIDE": "32",
}.items():
    os.environ.setdefault(k, v)

from train_mamba3_hybrid import Hyperparameters, GPT

args = Hyperparameters()
device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Build model
model = GPT(
    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap,
    use_smeargate=args.use_smeargate, use_bigram_hash=args.use_bigram_hash,
    bigram_buckets=args.bigram_buckets, bigram_hash_dim=args.bigram_hash_dim,
    use_ortho_init=args.use_ortho_init,
    mamba3_d_state=args.mamba3_d_state, mamba3_expand=args.mamba3_expand,
    mamba3_headdim=args.mamba3_headdim, mamba3_chunk_size=args.mamba3_chunk_size,
    mamba3_ngroups=args.mamba3_ngroups, mamba3_rope_fraction=args.mamba3_rope_fraction,
    mamba3_outproj_norm=args.mamba3_outproj_norm,
    num_attn_layers=args.num_attn_layers, num_heads=args.num_heads,
    num_kv_heads=args.num_kv_heads, rope_base=args.rope_base,
    qk_gain_init=args.qk_gain_init,
    ve_enabled=args.ve_enabled, ve_dim=args.ve_dim,
).to(device).bfloat16()

print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# Fake batch
seq_len = args.train_seq_len
bsz = 131072 // seq_len
x = torch.randint(0, args.vocab_size, (bsz, seq_len), device=device)
y = torch.randint(0, args.vocab_size, (bsz, seq_len), device=device)
print(f"Batch: {bsz} seqs x {seq_len} tokens = {bsz * seq_len} tokens/step")


def bench_and_profile(model, label, trace_name, warmup=20, bench=50, profile_steps=5):
    """Warmup, benchmark wall time, then profile."""
    model.train()

    # Warmup
    print(f"\n{'='*60}")
    print(f"[{label}] Warming up ({warmup} steps)...")
    for i in range(warmup):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()

    # Benchmark wall time
    print(f"[{label}] Benchmarking ({bench} steps)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(bench):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms_per_step = elapsed / bench * 1000
    print(f"[{label}] {ms_per_step:.1f} ms/step ({bench} steps, {elapsed:.2f}s total)")

    # Profile
    print(f"[{label}] Profiling ({profile_steps} steps)...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(profile_steps):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            loss.backward()
            model.zero_grad()
            torch.cuda.synchronize()

    prof.export_chrome_trace(f"profiling/{trace_name}")
    print(f"[{label}] Trace saved: profiling/{trace_name}")

    print(f"\n[{label}] TOP 20 SELF CUDA TIME")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

    return ms_per_step


# --- Baseline (no compile) ---
ms_baseline = bench_and_profile(model, "baseline", "trace_baseline.json")

# --- Compiled model ---
print("\n\nCompiling model with torch.compile...")
compiled_model = torch.compile(model)
ms_compiled = bench_and_profile(compiled_model, "compiled", "trace_compiled.json")

# --- Summary ---
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Baseline:  {ms_baseline:.1f} ms/step")
print(f"Compiled:  {ms_compiled:.1f} ms/step")
print(f"Speedup:   {ms_baseline / ms_compiled:.2f}x ({ms_baseline - ms_compiled:.1f} ms saved)")
