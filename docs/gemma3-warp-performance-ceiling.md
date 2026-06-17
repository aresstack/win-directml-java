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
| 15 | fused post-attn/post-ff RMSNorm + residual add (norm+add → 1) | dispatches **380 → 344**, uav barriers **399 → 363**; decode flat (WARP ~600 ms, GPU ~210 ms) |
| 16 | fused QKV + GateUp projection groups (3→1 q/k/v, 2→1 gate/up DML-GEMMs) | dispatches **344 → 290**, uav barriers **363 → 309**; **WARP decode ~600 → ~525 ms (~15%)**, GPU ~210 → ~150 ms |
| 17 | projection-fusion hardening + measured per-group timing breakdown (no new optimization) | parity/edge tests; **WARP decode is matmul-bound: lm-head 52%, the 5 GEMMs ~90%, all element kernels ~10%** |
| 18 | LM-head DML-GEMM shape probe (measure only) | **baseline GEMV (265 ms) is optimal**; GEMM M=1 3.3× slower, row-blocking neutral/worse → WARP at its fp32 DML-GEMM compute ceiling |

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

## 6. GEMMA-AUTO-GPU-1 result — WARP-CPU vs hardware GPU (measured)

Done. The adapter is now decoupled from the matvec path: `WindowsBindings.AdapterMode` (`WARP` /
`HARDWARE` / `DEFAULT`) selects the DXGI adapter explicitly while the backend stays `directml` (DML-GEMM
matvec, per the 14a decision). `Backend = WARP` → explicit WARP software adapter; `Backend = AUTO` → first
hardware adapter (software/WARP skipped), or a clear `"No hardware DirectML adapter found; use Backend=WARP
or configure -Dwindirectml.dxgi.adapterIndex."` if none. Gemma+AUTO now routes to the **same native
DirectML runtime** on the hardware adapter — no external-Python on the normal Gemma product path.

Measured on `WHERETHEHEARTIS` (gated `Gemma3AutoGpuProfileTest`, `-Dgemma.warp.realModel=true
-Dgemma.auto.gpu=true`), real `gemma-3-270m-it`, 16 output tokens. Adapters resolved: WARP = *Microsoft
Basic Render Driver* (0x1414/0x008C, software); HARDWARE = *NVIDIA GeForce RTX 5080* (0x10DE/0x2C02).

| prompt | adapter | prefill ms | decode total ms | decode avg/token | submits/tok | dispatches/tok | uav barriers/tok |
|--------|---------|-----------:|----------------:|-----------------:|------------:|---------------:|-----------------:|
| A (6 tok)   | WARP     | 6310  | 8913 | **594.2** | 21.0 | 380 | 399 |
| A (6 tok)   | HARDWARE | 2149  | 3261 | **217.4** | 21.0 | 380 | 399 |
| B (19 tok)  | WARP     | 3943  | 9080 | **605.3** | 23.4 | 380 | 399 |
| B (19 tok)  | HARDWARE | 4463  | 3215 | **214.3** | 23.4 | 380 | 399 |
| C (73 tok)  | WARP     | 9436  | 9079 | **605.3** | 21.0 | 380 | 399 |
| C (73 tok)  | HARDWARE | 14679 | 3290 | **219.3** | 21.0 | 380 | 399 |

Both adapters produce identical Top-1 " Paris" for "The capital of France is" (Paris smoke green for WARP
and HARDWARE — no silent wrong output).

**Findings:**
- **Decode is ~2.7–2.8× faster on the GPU** (~217 ms/token vs ~600 ms/token) with **identical**
  submit/fence/readback/dispatch/uav-barrier counts. This directly confirms §2: the WARP decode cost is the
  fixed per-dispatch/per-barrier overhead of ~380 tiny dispatches on the **CPU** rasterizer, not an
  algorithmic or submit-count problem. The same code, same counts, just runs each dispatch far cheaper on
  real hardware.
- **Prefill is mixed**: GPU wins the short prompt (A) but the GPU's first long batched prefill (C, 73 tok)
  is *slower* than WARP here (14.7 s vs 9.4 s) — consistent with one-time GPU shader compilation / cold
  batched-prefill cost paid on first hardware use (the HARDWARE leg runs cold after the WARP leg). Steady
  decode, the metric that matters per token, is the clean ~2.7× GPU win.

**Updated recommendation:** the hardware GPU **is** the product path for native Gemma when a hardware
adapter is present (`Backend = AUTO`). WARP stays the correct, identical-output fallback for hosts without a
GPU. Broad WARP-CPU kernel-fusion work (options A–B) remains not worth it: it cannot beat moving the
unchanged dispatch stream onto real hardware.

## 7. GEMMA-WARP-15 — per-layer dispatch breakdown + one more fusion (measured)

Done. First a per-decode-token, per-layer dispatch breakdown (`Gemma3DecodeDispatchBreakdown`, validated
against the empirical 380/token), then the single largest remaining fusion candidate.

**Decode dispatch breakdown — 21 dispatches/layer (×18) + 2 tail = 380/token (pre-15):**

| group | per layer | per token | kind |
|-------|----------:|----------:|------|
| q/k/v/o projection | 4 | 72 | DML-GEMM (kept) |
| mlp gate/up/down | 3 | 54 | DML-GEMM (kept) |
| rmsnorm (input, pre-ff) | 2 | 36 | small WARP (feeds a matvec — not fuseable) |
| **rmsnorm post-attn/post-ff** | **2** | **36** | **small WARP — fused with the add below** |
| **residual element-add** | **2** | **36** | **small WARP — fused with the norm above** |
| qk-norm (q, k) | 2 | 36 | small WARP |
| rope (q, k) | 2 | 36 | small WARP |
| kv-append (k, v) | 2 | 36 | small WARP |
| fused attention | 1 | 18 | small WARP (already fused in 14b) |
| geglu | 1 | 18 | small WARP |
| tail: final rmsnorm + lm-head | — | 2 | 1 small WARP + 1 DML-GEMM |

The 7 DML-GEMM matvecs/layer are kept (14a). Of the 14 small WARP dispatches/layer, the **largest fuseable
group is the post-norm + residual-add chain**: Gemma post-norms each sublayer output, then adds it to the
residual (`out = residual + rmsnorm(x, w)`), which ran as two dispatches (RMSNorm, then element-add) with a
UAV barrier between — at both the post-attention and post-feedforward points (4 dispatches/layer combined).

**Chosen candidate (only this one):** fuse RMSNorm + residual-add into one kernel
(`Gemma3WarpFusedNormAddKernel`, `ZERO_CENTERED_RMSNORM_ADD_HLSL`) — the residual add folds into the norm's
final store loop, fully parallel, no extra serial work (unlike 14b's attention fusion). Used at the two
post-norm sites in `decodeStepResident` and `forwardPrefillBatched`. Byte-identical to the two-kernel path
(unit-tested vs the CPU reference). DML-GEMM, INT4, the .wdmlpack format and the qk-norm/rope/kv groups are
untouched.

**Measured (`Gemma3AutoGpuProfileTest`, real `gemma-3-270m-it`, 16 output tokens):**

| metric | before (14b) | after (15) |
|--------|-------------:|-----------:|
| dispatches/token | 380 | **344** (−36 = 2/layer×18) |
| uav barriers/token | 399 | **363** (−36) |
| submits/token | 21 | 21 |
| fence waits/token | 21 | 21 |
| readbacks/token | 1 | 1 |
| decode avg/token (WARP) | ~600 ms | ~600–660 ms (flat/noisy) |
| decode avg/token (RTX 5080) | ~217 ms | ~200–230 ms (flat/noisy) |

Top-1 " Paris" (token 9079) identical for WARP and HARDWARE — Paris smoke green.

**Finding (reinforces 14b):** the fusion removes the predicted 36 dispatches + 36 barriers/token (≈9.5%)
and 2 scratch buffers/layer, but **decode wall-clock stays flat within noise** on both WARP and the GPU.
Dispatch/barrier *count* is no longer the lever — on WARP the per-dispatch CPU-rasterizer compute dominates;
on the GPU 344 small dispatches/token is already cheap and not the bottleneck. This closes out broad
small-kernel fusion as a WARP wall-clock lever (count is now lean: 7 matvecs + 12 small WARP dispatches +
tail). Further WARP-CPU decode speedups would need a structurally different approach (e.g. a single
per-layer megakernel — high risk, uncertain payoff per §3); the GPU path (`Backend = AUTO`) remains the
real product lever.

## 8. GEMMA-WARP-16 — fuse the projection groups (measured)

Done. Where §7 fused *small* element kernels (count down, wall-clock flat), GEMMA-WARP-16 reduces the
*large* DML-GEMM projection groups: the per-layer q/k/v projections become **one fused QKV DML-GEMM**
(`[attnDim+2·kvDim, hidden]`, output sliced into Q/K/V zero-copy views) and gate/up become **one fused
GateUp DML-GEMM** (`[2·intermediate, hidden]`, its `[gate|up]` output feeds the existing single-buffer
GeGLU directly). DML-GEMM stays — no custom FP32/INT4 matvec, no INT4 rebuild, no `.wdmlpack` change. The
fused weights are packed at runtime load only (`WarpDenseProjection.fromFusedWeightSources` →
`MatMulNBitsKernel.fromFusedFp32ByteBuffers`, heap-light when the source has an FP32 LE slice); the file
format is untouched. The output slices use a new non-owning `WarpGpuBuffer.slice` view (base VA + 4-byte
aligned offset), so there is **no extra copy** to split the fused output. Byte-identical to separate
projections (same per-row dot products); the float[] oracle paths slice the output too.

Per layer: q/k/v (3 matmuls → 1) + gate/up (2 → 1) = **3 fewer dispatches/layer × 18 = 54 fewer/token**.

**Measured (`Gemma3AutoGpuProfileTest`, real `gemma-3-270m-it`, 16 output tokens):**

| metric | before (15) | after (16) |
|--------|------------:|-----------:|
| dispatches/token | 344 | **290** (−54 = 3/layer×18) |
| uav barriers/token | 363 | **309** (−54) |
| submits/token | 21 | 21 |
| fence waits/token | 21 | 21 |
| readbacks/token | 1 | 1 |
| decode avg/token (WARP) | ~600–660 ms | **~524–533 ms (~15–20% faster)** |
| total gen ms (WARP) | A 15339 / B 16775 / C 23287 | **A 13189 / B 13509 / C 17031** |
| decode avg/token (RTX 5080) | ~200–230 ms | ~144–154 ms |

QKV fused active=**true**, GateUp fused active=**true**. Top-1 " Paris" (token 9079) identical for WARP and
HARDWARE — Paris smoke green.

**Finding (the real WARP lever):** unlike §7, fusing the *large* projection groups **does** move WARP
wall-clock (~15–20% faster decode, and prefill/total drop further from the batched fused projections).
Fewer, larger GEMMs read the shared hidden state once and use the WARP CPU rasterizer's SIMD/cache far
better than several small matmuls — and each fused matvec saves its staging copies + UAV barrier. This is
the structural win small-kernel fusion could not deliver: the lever on WARP is **GEMM shape/count**, not
element-dispatch count. Remaining matmuls/layer are now 4 (QKV, O, GateUp, Down) + the tied LM head. The
GPU path benefits too (~30% faster decode) and remains the optional accelerator (`Backend = AUTO`).

## 9. GEMMA-WARP-17 — hardening + measured per-group timing (no new optimization)

Two deliverables, no new optimization. (1) Hardened the GEMMA-WARP-16 projection fusion as the product
path; (2) measured where the WARP decode token time actually goes, per kernel group, to choose the next
lever from data.

**Hardening.** Added explicit parity (`WarpDenseProjectionFusionParityTest`: a fused projection == the
separate parts, for the 3-part QKV and 2-part GateUp shapes) and fail-fast guards (mismatched input width
or a wrong-sized part throws — never a silent wrong fused matrix). `WarpGpuBuffer.slice` edge cases
(`WarpGpuBufferSliceTest`: bounds/offset, the 4-byte-aligned VA offset, no re-slice/readback on a view,
non-owning `close()`, and a functional UAV-binding check). `Gemma3WarpMlp.mlpBatched` now falls back to the
per-position fused MLP when batched matmul is unavailable or the row cap is exceeded (symmetric with the
QKV batched path) — no silent mis-shaped matmul. QKV/GateUp fusion stays active.

**Measured per-group timing.** New opt-in `WarpGroupProfiler` + `WarpExecutionContext.mark/endGroup`
(`-Dgemma.warp.realModel=true -Dgemma.warp.groupProfile=true`, `Gemma3WarpDecodeGroupProfileTest`). The
normal runtime is untouched (no sink attached → `mark()` is a no-op, coalescing/the deferred batch are
unchanged). In profile mode the decode keeps coalescing but drops the deferred batch, so each `mark()`
boundary flushes+fences that group's command list once and times it — one submit/fence per group,
representative of the coalesced pipeline (profiled total ≈ real coalesced decode, e.g. 531 ≈ 522 ms/token
on WARP, confirming the attribution is accurate).

Measured on the **explicit WARP software adapter** (the CPU-only product path), real `gemma-3-270m-it`,
warm decode tokens:

| group | dispatches/token | ms/token | % of decode |
|-------|-----------------:|---------:|------------:|
| **lm-head** (tied, 262144×640) | 1 | ~278 | **52.3%** |
| **gate+up projection** | 18 | ~90 | **17.0%** |
| qkv projection | 18 | ~41 | 7.8% |
| down projection | 18 | ~40 | 7.6% |
| o projection | 18 | ~28 | 5.2% |
| qk-norm + rope + kv-append | 108 | ~15 | 2.8% |
| attention-context | 18 | ~8 | 1.5% |
| input / preff / post-attn / postff norms | 18 ea | ~6 ea | ~1.1% ea |
| geglu | 18 | ~6 | 1.1% |
| final-rmsnorm + logits-readback + token-selection | — | ~2 | ~0.3% |

Paris smoke green on WARP (token 9079).

**Finding — WARP decode is matmul-bound, and LM-head-bound:** the five GEMMs are **~90%** of the WARP decode
token, the tied **LM head alone is 52%** (a 262144×640 FP32 matvec run every token on the CPU rasterizer),
and **all element/attention kernels together are only ~10%**. This is the same adapter the §7/§8 work runs
on, and it explains both prior results cleanly: §7's small-kernel fusion was flat because those kernels are
a tenth of the time; §8's GEMM fusion was a real win because GEMMs are nine-tenths. (For contrast, on the
hardware GPU the distribution is flat ~7–8%/group and the LM head is ~1% — so this is specifically a WARP
finding.)

**Recommendation for GEMMA-WARP-18 (measure, do not blindly build):**
- *Biggest remaining lever:* the **tied LM head (52%)**. It is a `[262144, 640]·[640]` FP32 matvec every
  token. Investigate, data-first, whether a better DML-GEMM shape/path helps on WARP — e.g. the batched/
  tiled matmul config, or row-blocking the 262144 output — keeping DML-GEMM (no custom/INT4 kernel, no
  `.wdmlpack` change). Greedy decode still needs all logits (argmax), so the matmul itself is the target,
  not skipping outputs.
- *Second lever:* the **MLP GEMMs (GateUp 17% + Down 7.6% ≈ 25%)** — check the GateUp/Down GEMM shapes on
  WARP (the intermediate=2048 matmuls dominate the per-layer cost).
- *Do not pursue:* further **small-element fusion** (qk-norm/rope/kv-append 2.8%, each norm/geglu ~1%) — it
  cannot move WARP wall-clock (confirms §7); and the **attention/KV data path** (1.5%) is negligible on WARP.

WARP stays the product path; AUTO/GPU stays the optional control path.

## 10. GEMMA-WARP-18 — LM-head DML-GEMM shape probe (measured; result: keep baseline)

Done, measurement only — the product LM-head path is unchanged. The §9 lever (the tied LM head, 52% of WARP
decode) was probed for a better DML-GEMM shape. `Gemma3WarpLmHeadShapeProbeTest`
(`-Dgemma.warp.realModel=true -Dgemma.warp.lmHeadProbe=true`), explicit WARP adapter, real model, matmul-only
timing (warm + averaged), every variant fed the same input and verified to keep the full-vocab Top-1; the
product session separately confirms " Paris" (9079).

| variant | layout | dispatches | lm-head ms/token | decode est ms | mem | Top-1 |
|---------|--------|-----------:|-----------------:|--------------:|-----|------:|
| **baseline-matvec (GEMV)** | `[262144,640]·[640]` | 1 | **265.5** | 506 | 640 MB | = |
| batched-m1 (GEMM M=1) | `[1,640]·[262144,640]ᵀ` | 1 | 874.0 | 1115 | 640 MB | = |
| rowblock-4-matvec | 4×`[65536,640]` coalesced | 4 | 267.5 | 508 | 640 MB | = |
| rowblock-16-matvec | 16×`[16384,640]` coalesced | 16 | 281.2 | 522 | 640 MB | = |

(real decode ≈ 506 ms/token; rest-of-decode ≈ 241 ms; all variants identical Top-1; Paris green.)

**Findings:**
1. **Which is fastest?** The **current baseline GEMV (`matvecResident`, 265 ms)** — nothing beats it.
2. **Gain?** None. The GEMM M=1 form is **3.3× slower** (874 ms — the batched shader is not tuned for a
   single row), and row-blocking is **neutral to worse** (4 blocks +0.7%, 16 blocks +5.9% from per-dispatch
   overhead).
3. **Extra memory?** None — all variants are the same 640 MB tied weights (row-blocking just slices them; 0
   overhead, but also 0 speedup).
4. **Production-ready variant?** No — none is worth productizing.
5. **GEMMA-WARP-19?** Do **not** change the LM head. It is already optimal in DML-GEMM form.

**Ceiling note:** the LM head is ~168M MACs/token at ~265 ms ≈ **0.6 GMAC/s** effective; the MLP GEMMs sit
at the same throughput (~70M MACs/130 ms). WARP decode is now at its **fp32 DML-GEMM compute ceiling** on the
CPU rasterizer — the time is raw FLOPs, not shape/overhead. Within the allowed constraints (DML-GEMM, fp32,
no `.wdmlpack`/INT4/custom-kernel change) there is no faster LM head, and the MLP GEMMs face the same wall.
The only structural levers left are out of scope here (quantization to cut FLOPs, or the optional GPU path).
**Recommendation: LM head stays as-is; treat WARP decode as at its fp32 compute ceiling. Any further WARP
speedup needs a FLOP reduction (quantization) — a separate, larger decision — otherwise the hardware GPU
(`Backend = AUTO`) remains the optional accelerator.**
