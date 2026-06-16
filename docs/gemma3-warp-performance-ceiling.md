# Gemma 3 270M native runtime — performance ceiling & WARP-vs-GPU decision (GEMMA-WARP-14c)

Decision document, not a new optimization. Captures where the native Gemma runtime stands after the
GEMMA-WARP-13*/14* perf work, what the current bottleneck is, which WARP-CPU optimizations remain, and why
the next real lever is a hardware-GPU comparison rather than more small-kernel fusion. No runtime/default
change in this slice. Companion to the running log in `gemma3-warp-runtime-plan.md`.

All numbers below are measured on this dev host (`WHERETHEHEARTIS`) for the real `gemma-3-270m-it` model
via the gated tests (`-Dgemma.warp.realModel=true`, profile via `-Ddirectml.generation.profile=true`).
No invented figures.

## 1. Optimization history (native WARP/DirectML resident path)

| slice | change | key measured effect (decode/token unless noted) |
|-------|--------|--------------------------------------------------|
| 13b-1 | submit/fence/readback instrumentation | baseline ≈ **1140 submits**, **344 readbacks** |
| 13b-2 | GPU-resident buffer/execution-context seam | attn-chain readbacks 4→1, parity |
| 13b-3a | resident decodeStep wiring | readbacks **344 → 37**, submits 1140 → 834 |
| 13b-3b | submit/fence coalescing (DirectMlGpuBatch deferred fence) | fence waits **834 → 93**, submits 834 → 454 |
| 13b-4 | resident/batched made the native-warp product path | (no counter change; product path switch) |
| 13c | GPU-resident KV cache (no per-layer k/v readback) | readbacks **38 → 1**, fence waits **97 → 21** |
| 13d | command-list coalescing (UAV dispatches share one list) | submits **418 → 220**, fences 21, decode ≈ 442 → 294 ms |
| 13e | batched prefill (whole prompt in one pass) | prefill submits 1417 → 261 (6-tok); decode unchanged |
| 13f | cold/warm profiling + shader warm-up | warm decode ≈ 250 ms/token; prefill submits ≈ const in prompt len |
| 13g | matvec projections recorded into the layer list | submits **220 → 21**; decode ≈ 250 ms **unchanged** |
| 14a | matvec compute benchmark + decision | DML-GEMM ≈ **20 ms** matvec/token vs custom WARP-FP32 ≈130 / INT4 ≈145 → keep DML |
| 14b | fused attention (scores+softmax+value → 1) + dispatch/barrier counters | dispatches **416 → 380**, uav barriers **435 → 399**; decode flat |

Consolidated per-decode-token trajectory (real model):

```
readbacks/token:     344 → 37 → 1
fence waits/token:   834 → 93 → 21
submits/token:       1140 → 834 → 454 → 220 → 21
dispatches/token:    416 → 380
uav barriers/token:  435 → 399
decode avg/token:    ~452 ms (pre-13d) → ~294 ms (13d) → ~210–280 ms (14b, noisy)
```

## 2. Current bottleneck

The native runtime is **no longer** bound by any of the things the 13*/14* slices fixed:

- not Python (the WARP path is pure Java/DirectML),
- not JVM heap (heap-light weight load),
- not host KV-cache round-trips (GPU-resident KV cache, ~1 readback/token),
- not D3D12 submits/fences/readbacks (21 / 21 / 1 per token),
- not MatVec/GEMM (DML-GEMM ≈ 20 ms of the ~250 ms decode token — 14a).

It **is** bound by the **per-dispatch / per-UAV-barrier cost of ~380 tiny element/attention dispatches per
token on the WARP software rasterizer**. 14b confirmed this is a *fixed-overhead* wall, not a flop wall:
fusing 3 attention dispatches into 1 cut dispatches 416→380 and barriers 435→399 but left wall-clock flat
(the fused kernel does the same work in one heavier, less-parallel dispatch).

### Why it is "WARP" at all
`Gemma3NativeWarpRuntime` initializes the **`directml`** backend, i.e. the *default* DXGI adapter
(`enumAdapters1`, index 0) — **not** an explicit WARP adapter. On this host the only/default adapter is the
**WARP software rasterizer** (a CPU implementation of D3D12), so every dispatch pays a fixed CPU launch +
barrier cost. The "native WARP" name is therefore about *where this machine runs it*, not a hard-coded
software path: on a machine with a hardware GPU at adapter 0 (or via `-Dwindirectml.dxgi.adapterIndex=N`),
the **same** native code would dispatch to the GPU, where ~380 small dispatches/token is cheap.

## 3. Remaining WARP-CPU optimizations (realistic assessment)

| option | what | assessment |
|--------|------|------------|
| A) more kernel fusion | residual+RMSNorm pairs, QK-norm+RoPE chain | lowers dispatch count further, but 14b showed that alone does **not** reliably lower WARP wall-clock; diminishing returns |
| B) one big fused per-layer kernel | collapse most of a layer into one shader | technically risky, harder to test, hurts numerics/debuggability; high effort, uncertain payoff on WARP |
| C) batched/multi-query attention for long prefill | one attention dispatch over all query positions | worthwhile for long prompts (prefill is O(seqLen) per-position attention today), but a sizable separate slice; helps prefill, not steady-state decode |
| D) INT4 / quantized weights | packed INT4 matvec | not compelling: 14a measured custom INT4-WARP **slower** than DML-GEMM, and Gemma 270M ships FP32 (no clean quant path) |

None of A–D is a clear win on the WARP CPU rasterizer; A/B specifically fight a fixed per-dispatch overhead
that fusion only partially removes.

## 4. AUTO / hardware-GPU comparison — the next real lever

The biggest unknown is simply **the adapter**: all numbers above are the WARP CPU rasterizer. The native
code path is already adapter-agnostic (it inits `directml` = default adapter), so measuring it on a hardware
GPU is the highest-value next step and needs little/no new code.

Proposed next slice **GEMMA-AUTO-GPU-1** (measurement):
- run the **same** native Java/WARP/DirectML Gemma codepath on a **hardware** DirectML adapter
  (a machine with a GPU, or `-Dwindirectml.dxgi.adapterIndex=N` selecting a hardware adapter), no Python,
- capture the **same** profile metrics (prefill/decode ms, submits/fences/readbacks, dispatches/uav barriers
  per token),
- compare WARP-CPU vs hardware-GPU decode avg/token to quantify the ceiling that fusion cannot move.

### Open implementation point (NOT changed in this slice)
In the Workbench, Gemma runs the native path only when **Backend = WARP**
(`SummarizerPanel.gemmaUsesNativeWarp(backend) == (backend == Backend.WARP)`); **Backend = AUTO** currently
falls back to the **external Python/Transformers** probe. For GEMMA-AUTO-GPU-1, `Backend = AUTO` should
select the **native** Gemma DirectML runtime on a hardware adapter when available (not external-python).
That rewiring is deferred to GEMMA-AUTO-GPU-1; this slice only documents it.

## 5. Recommendation

**Do not spend more broad effort on small WARP-CPU kernel fusion before measuring on a hardware GPU.**

Rationale: submits/fences/readbacks and MatVec are solved; the remaining decode cost is the fixed
per-dispatch/per-barrier overhead of ~380 tiny dispatches on the WARP CPU rasterizer, which 14b showed
fusion does not reliably reduce in wall-clock. The same native code already targets the default DirectML
adapter, so a hardware GPU is expected to collapse that overhead with no algorithmic change.

**Next step: GEMMA-AUTO-GPU-1** — measure the native Gemma runtime on a hardware/AUTO adapter and compare to
WARP. Revisit further WARP-CPU fusion (options A–C) only if a hardware GPU is unavailable and WARP must
remain the target.
