"""
Benchmarks kernel fusion for mamba3 backward to ensure correctness and
performance benefits.

The stock kernel uses atomic_add reductions in several places, so two runs
with identical inputs produce slightly different grads. We quantify that
noise floor by saving two stock references and comparing them, then use a
2x-noise-floor relative-L2 threshold to tell a real fusion bug apart from
atomic-add jitter.

Usage:
    # 1) Capture two independent stock references (same code, different runs)
    python3 triton_kernels/bench_mamba3_bwd.py --save triton_kernels/ref_a.pt
    python3 triton_kernels/bench_mamba3_bwd.py --save triton_kernels/ref_b.pt

    # 2) Measure the noise floor (per-tensor relative L2 between the two refs)
    python3 triton_kernels/bench_mamba3_bwd.py \
        --noise-floor triton_kernels/ref_a.pt triton_kernels/ref_b.pt

    # 3) Run the fused kernel and compare against one reference
    MAMBA3_FUSED_BWD=1 python3 triton_kernels/bench_mamba3_bwd.py \
        --check triton_kernels/ref_a.pt --l2-tol 0.05
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

import torch

from train_mamba3_hybrid import Hyperparameters, Block

parser = argparse.ArgumentParser()
parser.add_argument("--save", type=str, default=None,
                    help="Save reference grads to this path (typically used with stock kernel)")
parser.add_argument("--check", type=str, default=None,
                    help="Compare computed grads against reference at this path (relative L2)")
parser.add_argument("--noise-floor", type=str, nargs=2, default=None,
                    metavar=("REF_A", "REF_B"),
                    help="Report per-tensor relative L2 between two stock references. "
                         "Use two independent --save runs to quantify atomic_add jitter.")
parser.add_argument("--l2-tol", type=float, default=0.05,
                    help="Relative L2 tolerance for --check (||g-r||/||r||). "
                         "Rule of thumb: ~2x the noise floor from --noise-floor.")
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--iters", type=int, default=50)
cli = parser.parse_args()


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """||a - b||_2 / ||b||_2 in fp32. Returns inf if b is all zero and a isn't."""
    af = a.float()
    bf = b.float()
    num = (af - bf).norm().item()
    den = bf.norm().item()
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


def compare_grad_dicts(got: dict, ref: dict, tol: float, label: str):
    """Relative-L2 compare; returns (n_ok, n_fail, worst_rel, worst_name)."""
    n_ok = n_fail = 0
    worst_rel = 0.0
    worst_name = ""
    for name, g in got.items():
        if name not in ref:
            print(f"  MISSING in ref:  {name}")
            n_fail += 1
            continue
        r = ref[name]
        if g.shape != r.shape:
            print(f"  SHAPE MISMATCH {name}: {tuple(g.shape)} vs {tuple(r.shape)}")
            n_fail += 1
            continue
        rel = rel_l2(g, r)
        if rel > worst_rel:
            worst_rel, worst_name = rel, name
        if rel <= tol:
            n_ok += 1
        else:
            diff = (g.float() - r.float()).abs()
            print(f"  {label} {name}: "
                  f"rel_l2={rel:.2e}  "
                  f"max_abs={diff.max().item():.2e}  "
                  f"shape={tuple(g.shape)}")
            n_fail += 1
    return n_ok, n_fail, worst_rel, worst_name

device = torch.device("cuda")

# --- Noise-floor mode: compare two stock refs and exit, no model needed ---
if cli.noise_floor is not None:
    ref_a_path, ref_b_path = cli.noise_floor
    ref_a = torch.load(ref_a_path, map_location=device)
    ref_b = torch.load(ref_b_path, map_location=device)
    print(f"Noise floor: {ref_a_path}  vs  {ref_b_path}")
    print(f"{'tensor':<32} {'rel_l2':>10} {'max_abs':>12} {'shape'}")
    worst = 0.0
    worst_name = ""
    for name in sorted(ref_a.keys()):
        if name not in ref_b:
            print(f"  MISSING in B: {name}")
            continue
        a, b = ref_a[name], ref_b[name]
        rel = rel_l2(a, b)
        diff = (a.float() - b.float()).abs().max().item()
        print(f"{name:<32} {rel:>10.2e} {diff:>12.2e}  {tuple(a.shape)}")
        if rel > worst:
            worst, worst_name = rel, name
    print()
    print(f"Worst relative L2: {worst:.2e}  ({worst_name})")
    print(f"Suggested --l2-tol for --check: {max(2*worst, 1e-4):.2e}")
    sys.exit(0)

args = Hyperparameters()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Build a Block then isolate the Mamba3Layer for clean perf attribution.
# Block wraps Mamba3Layer + MLP + norms + residuals; we only want the Mamba3 backward.
# Seed BEFORE construction so model weights are identical across runs — otherwise
# --save runs produce uncorrelated grads (not atomic_add noise) and --check is meaningless.
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
mamba_block = Block(
    args.model_dim, args.mlp_mult,
    args.mamba3_d_state, args.mamba3_expand,
    args.mamba3_headdim, args.mamba3_chunk_size,
    args.mamba3_ngroups, args.mamba3_rope_fraction,
    args.mamba3_outproj_norm,
).to(device).bfloat16()

m3 = mamba_block.mamba3
m3.train()

seq_len = args.train_seq_len
bsz = 131072 // seq_len  # same per-GPU micro-batch as real training

print(f"Model:  {sum(p.numel() for p in m3.parameters())/1e6:.2f}M params (Mamba3Layer only)")
print(f"Batch:  {bsz} seqs x {seq_len} = {bsz * seq_len} tokens")
print(f"Fused:  MAMBA3_FUSED_BWD={os.environ.get('MAMBA3_FUSED_BWD', '0')}")
print()

def zero_grads(inp):
    if inp.grad is not None:
        inp.grad = None
    for p in m3.parameters():
        p.grad = None

# ---- Warmup: covers Triton autotune + compile ----
torch.manual_seed(0)
x_warm = torch.randn(bsz, seq_len, args.model_dim,
                     device=device, dtype=torch.bfloat16, requires_grad=True)
print(f"Warmup ({cli.warmup} iters)...")
for _ in range(cli.warmup):
    zero_grads(x_warm)
    out = m3(x_warm)
    loss = out.float().pow(2).mean()
    loss.backward()
torch.cuda.synchronize()

# ---- Deterministic reference capture ----
# Fresh seeded tensor so --save and --check runs produce identical inputs.
torch.manual_seed(42)
x_ref = torch.randn(bsz, seq_len, args.model_dim,
                    device=device, dtype=torch.bfloat16, requires_grad=True)
zero_grads(x_ref)
out = m3(x_ref)
loss = out.float().pow(2).mean()
loss.backward()

ref_grads = {"__input__": x_ref.grad.detach().clone()}
for name, p in m3.named_parameters():
    if p.grad is not None:
        ref_grads[name] = p.grad.detach().clone()
print(f"Captured {len(ref_grads)} gradient tensors")

if cli.save:
    torch.save(ref_grads, cli.save)
    print(f"Saved reference grads -> {cli.save}")

if cli.check:
    ref = torch.load(cli.check, map_location=device)
    n_ok, n_fail, worst_rel, worst_name = compare_grad_dicts(
        ref_grads, ref, cli.l2_tol, label="MISMATCH")
    print(f"Correctness: {n_ok} OK, {n_fail} FAIL  "
          f"(l2_tol={cli.l2_tol:.2e}, worst={worst_rel:.2e} on {worst_name})")
    if n_fail > 0:
        sys.exit(1)

# ---- Performance measurement ----
print(f"\nBenchmark ({cli.iters} iters)...")
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(cli.iters):
    zero_grads(x_warm)
    out = m3(x_warm)
    loss = out.float().pow(2).mean()
    loss.backward()
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / cli.iters * 1000
print(f"fwd+bwd: {ms:.2f} ms/iter  (batch={bsz}, seq={seq_len})")
