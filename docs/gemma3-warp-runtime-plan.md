# Gemma 3 270M-it — native Java/WARP runtime: progress & open points

Running log for the GEMMA-WARP-* slices (see `directml-inference/docs/gemma-3-270M.md` for the plan).
Each slice = own commit. Blockers are recorded here and the next buildable step is taken instead of
stopping. Open points are resolved with the user at the end.

## Confirmed config (real, from `gemma-3-270m-it/config.json`)

| field | value |
|-------|-------|
| model_type / arch | `gemma3_text` / `Gemma3ForCausalLM` |
| hidden_size | 640 |
| intermediate_size | 2048 |
| num_hidden_layers | 18 |
| num_attention_heads | 4 |
| num_key_value_heads | 1 (GQA) |
| head_dim | 256 (decoupled: 4×256=1024 ≠ 640) |
| vocab_size | 262144 (256k) |
| max_position_embeddings | 32768 |
| sliding_window | 512 |
| _sliding_window_pattern | 6 (full attention at layers 5, 11, 17) |
| rope_theta (global) / rope_local_base_freq | 1_000_000 / 10_000 |
| query_pre_attn_scalar | 256 → attn scale 1/sqrt(256) |
| rms_norm_eps | 1e-6 (zero-centered RMSNorm) |
| hidden_activation | gelu_pytorch_tanh (GeGLU) |
| bos/eos/pad | 2 / 1 / 0 |
| softcapping | none (attn & final null) |
| tie_word_embeddings | (absent → Gemma ties embed↔lm_head) |

## Slice status

| Slice | Status |
|-------|--------|
| GEMMA-WARP-1 family shell + config/inspect | **done** (7b1dd3c) |
| GEMMA-WARP-2 tokenizer + chat template | pending |
| GEMMA-WARP-3 wdmlpack compiler shell | pending |
| GEMMA-WARP-4 CPU reference math | **done** (bd8fbb4) |
| GEMMA-WARP-8/9 reference weights + full forward | **done** (4b028fe) |
| GEMMA-WARP-9 real-model parity vs transformers | **done — PASS** (next token 9079 " Paris") |
| GEMMA-WARP-3 wdmlpack compiler + package | **done** (137f0f5) |
| GEMMA-WARP-7 attention layout + window mask (device-free) | **done** (0d83fbd) |
| GEMMA-WARP-10 reference greedy generation | **done** (036493b) |
| GEMMA-WARP-2 tokenizer + chat template | **done — exact transformers match** |
| GEMMA-WARP-5a WARP zero-centered RMSNorm + QK-Norm kernels | **done — GPU-validated vs reference** |
| GEMMA-WARP-6 WARP GeGLU + GELU-tanh MLP | **done — GPU-validated vs reference** |
| GEMMA-WARP-7a attention layout + RoPE reference (device-free) | **done** (layout 0d83fbd, RoPE this slice) |
| GEMMA-WARP-7b WARP RoPE + attention-scores primitives | **done — GPU-validated vs reference** |
| GEMMA-WARP-8/9 WARP single-layer / prefill | open — next |
| GEMMA-WARP-10 WARP decode session + KV cache | open — depends on WARP kernels |
| GEMMA-WARP-11 workbench native flag | open — depends on tokenizer + WARP |
| GEMMA-WARP-12 perf/heap comparison | open — depends on WARP |

## What works now (verified, device-free, no Python)

A complete **native CPU Gemma 3 runtime** in Java, numerically correct vs HF transformers:
`config.json` + `model.safetensors` → `Gemma3WdmlPackCompiler` → `.wdmlpack` →
`Gemma3RuntimePackage.loadReferenceWeights()` → `Gemma3ReferenceForwardPass` /
`Gemma3ReferenceGenerator` (greedy). Real-model parity PASSES (next token 9079 " Paris"). 24 gemma
tests green (1 gated parity ran locally). This CPU path is the **WARP parity oracle**, not the product
path.

**Milestone:** the device-free CPU reference is numerically correct against HF transformers
(`Gemma3RealModelParityTest`, gated): for "The capital of France is" (ids [2,818,5279,529,7001,563])
the Java reference greedy next token == transformers argmax == 9079 (" Paris"). The Gemma math
(zero-centered RMSNorm, QK-norm, dual-theta RoPE, GQA, sliding window, GeGLU, sandwich norms, tied LM
head) is therefore the trustworthy WARP parity oracle.

## Open points / blockers (resolve at the end)

- **GPU/WARP execution IS available on this host (`WHERETHEHEARTIS`).** The earlier "no device in the
  sandbox" assumption no longer holds for this machine: the existing `SmolLM2NativeWarpExecutorTest`
  runs (5 tests, 0 skipped, `backend=warp`), and the new `Gemma3WarpNormKernelTest` runs all 7 GPU
  cases. WARP-kernel slices are therefore validated on the real device, gated on
  `WindowsBindings.isSupported()` so they skip (not fail) on a CI box without an adapter. Strategy
  unchanged: the CPU reference (numerically correct vs HF/transformers) is the parity oracle; each
  WARP kernel is mirrored against it.

- **GEMMA-WARP-5a norm kernels — done, GPU-validated.** `Gemma3WarpNorms` (HLSL), `Gemma3WarpRmsNormKernel`
  (zero-centered RMSNorm over a whole vector) and `Gemma3WarpQkNormKernel` (per-head RMSNorm over
  `head_dim`, all heads in one dispatch). Both are the GPU mirror of `Gemma3ReferenceMath.rmsNormZeroCentered`
  and **scale by `(1 + weight)`** — kept strictly separate from the Phi-3/Qwen `RMSNORM_HLSL` (which
  scales by `weight`); those shaders are untouched. Validated on the real device against the verified
  CPU reference for small shapes, the real hidden=640 and head_dim=256 (incl. numHeads=1 GQA k-head),
  multiple eps, and the zero-weight identity-scale case. Tolerance **abs 1e-4 + rel 1e-4** (GPU float
  vs reference double sum-of-squares). `head_dim` is supplied by the caller, never derived as
  `hidden/heads`. Still validation building blocks (per-call upload/readback), not yet the fused
  product path.

- **GEMMA-WARP-6 GeGLU + GELU-tanh MLP — done, GPU-validated.** `Gemma3WarpActivations` (HLSL),
  `Gemma3WarpGeluTanhKernel`, `Gemma3WarpGeGluKernel` (fused `gelu_tanh(gate)*up`), and `Gemma3WarpMlp`
  composing the three matmuls over the **shared `WarpDenseProjection`** + the GeGLU kernel
  (`down_proj(gelu_tanh(gate_proj·x) * up_proj·x)`; the pre/post feedforward sandwich norms stay in the
  5a norm kernels, not in this block). Gemma's **GELU-tanh GeGLU** is kept strictly separate from the
  Qwen/SmolLM2 SiLU SwiGLU; the decoder-only SwiGlu kernels are untouched. Validated on the real device
  against the verified reference: GELU-tanh on signed/zero/tail values and a 2048-wide random vector,
  GeGLU on the real intermediate (incl. zero-gate→0), and the full MLP on a small shape and the real
  hidden=640/intermediate=2048 shape. Tolerance **abs 1e-4 + rel 1e-4** for the element-wise activation
  kernels; **abs 1e-3 + rel 1e-3** for the full MLP (three float matmuls accumulate in a different order
  than the reference dot). `MatMulNBitsKernel.fromDequantizedWeights` uploads the full FP32 matrix
  (no re-quantization), so this is a true FP32 parity — the real-shape test uses realistic small weight
  magnitudes (the regime a pre-feedforward-normed input actually produces). ByteBuffer norm-weight
  upload intentionally skipped (norm/activation vectors are tiny; heap-light matters for the big
  projections + tied embedding/LM-head).

- **GEMMA-WARP-7 RoPE + local/global attention layout — done.** Split into 7a (device-free) and 7b
  (WARP). **7a:** the local/global layout (`Gemma3AttentionLayout`) was already shipped in 0d83fbd
  (GQA mapping, full layers 5/11/17, dual theta, sliding-window + causal visibility); this slice adds
  the device-free `Gemma3RoPE` (rotate-half across heads, the parity oracle) plus `Gemma3RoPETest`
  rounding out the checklist (all-18-layer full/local classification, **explicit head_dim=256** — never
  `hidden/heads`, GQA with >1 kv head, window+causal). **7b:** `Gemma3WarpAttention` (HLSL) +
  `Gemma3WarpRoPEKernel` (rotate-half RoPE over packed heads) and `Gemma3WarpAttentionScoresKernel`
  (scaled masked `QK^T` for one query position; GQA `kvHead=head/groupsPerKv`; the visible range
  `[firstValid, queryPos]` comes from `Gemma3AttentionLayout`, so local/global + causal are applied
  without imitating any other family; masked keys get the `SCORE_SENTINEL = -1e30` a softmax drops).
  Validated on the real GPU vs the verified reference: RoPE on a small shape and the real head_dim=256
  at **both** thetas (1e4/1e6) and several positions (pos=0 identity); scores on a full layer
  (firstValid=0), a local layer with a biting sliding window, and a 4-heads/2-kv GQA case. Tolerance
  **abs 1e-4 + rel 1e-4**. Still primitives (per-call upload/readback), not yet the fused single-layer
  step — that is GEMMA-WARP-8 (softmax + AV + o_proj wiring).
- **GEMMA-WARP-2 Tokenizer — RESOLVED (done).** Native `Gemma3Tokenizer` reads `tokenizer.json` only
  (no `tokenizer.model`, no SentencePiece DLL, no Python): normalizer space→`▁`, BPE over the whole
  normalized string (the `Split(" ")` pre-tokenizer is a no-op post-normalization), byte_fallback to
  `<0xNN>`, added/special tokens carved out before BPE, `<bos>` prepended. Java ids match HF
  transformers **exactly** for all fixtures (English, German umlauts, Natural/ADABAS, Java, empty,
  unicode/byte-fallback) and the chat template; decode round-trips. Findings: **`tokenizer.json` alone
  suffices** (no `tokenizer.model` needed); byte-fallback **is** required (and implemented); metaspace
  normalizer correct; no leading-`▁` prepend at start. The full native chain
  **text → Java tokens → Java forward → Java decode = "Paris"** is verified (`nativeTextToTextProducesParis`).
  → A text→text workbench native path is now possible (pending the WARP runtime for speed).

- **WARP kernels (GEMMA-WARP-5/6/8/9/10, GPU-blocked here):** no DirectML/D3D12 device in this build
  sandbox, so on-GPU numerical validation cannot run (mirrors `SmolLM2NativeWarpExecutorTest`, gated on
  `WindowsBindings.isSupported()`). The CPU reference + `Gemma3AttentionLayout` give a ready parity
  oracle. Implementation plan when run on a Windows+GPU host: reuse `WarpDenseProjection`/
  `MatMulNBitsKernel` + the D3D12 ByteBuffer upload for q/k/v/o/gate/up/down and the LM head; add
  **new** Gemma kernels for zero-centered RMSNorm, QK-norm, GELU-tanh/GeGLU; drive attention from
  `Gemma3AttentionLayout` (local/global + sliding window + GQA + dual RoPE). Validate each kernel and
  the single-layer/prefill output against the CPU reference (tolerance documented), then add the WARP
  decode session + KV cache (local layers only need a windowed cache).

- **Heap (product path):** the reference materializes `embed_tokens` as FP32 (~671 MB for 270M) — fine
  for parity, not for the product path. The WARP runtime must be heap-light: ByteBuffer/mmap upload,
  and the tied `embed_tokens`/LM-head must not be uploaded twice. The 256k FP32 LM head dominates
  per-token cost (per the feasibility doc) — candidate for a later optimization slice.

- **Workbench native flag (GEMMA-WARP-11):** wire `-Dgemma.runtime=native-warp` to the native package
  path once tokenizer + WARP exist; keep `external-python-transformers` the default until WARP is
  proven on GPU. The external probe path stays intact (untouched this block).

- **GEMMA-WARP-12 perf:** pending a runnable WARP path; no fabricated numbers.
