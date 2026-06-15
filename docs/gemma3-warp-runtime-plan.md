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
| GEMMA-WARP-8 WARP single-layer (softmax + value + full attention) | **done — GPU-validated vs reference** |
| GEMMA-WARP-9 WARP full prefill (all layers + embedding + tied LM head) | **done — real model top-1 " Paris" on GPU** |
| GEMMA-WARP-10a WARP KV cache + single-token decodeNext | **done — real decode == full recompute on GPU** |
| GEMMA-WARP-10b WARP greedy generate + stop + streaming | **done — real " Paris." multi-step on GPU** |
| GEMMA-WARP-11 workbench native flag (-Dgemma.runtime=native-warp) | **done — native path " Paris." in the runner** |
| GEMMA-WARP-12 perf/heap measurement (native-warp vs external) | **done — measured; verdict WAIT** (see `gemma3-warp-runtime-performance.md`) |
| GEMMA-WARP-13a heap-light product weight load | **done — on-heap delta 0 MB at load, still " Paris"** |
| GEMMA-WORKBENCH-CONVERT-1 UI Convert → model_gemma3.wdmlpack | **done — Convert enabled + produces the package the runtime finds** |
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

- **GEMMA-WARP-8 single-layer — done, GPU-validated.** `Gemma3WarpSoftmaxKernel` (numerically-stable
  row-wise softmax; masked sentinels exp to 0) + `Gemma3WarpAttentionValueKernel`
  (`context = prob·V`, GQA `kvHead=head/groupsPerKv`) complete the attention chain; `Gemma3WarpLayer`
  composes the full Gemma layer from the validated blocks —
  `input RMSNorm → q/k/v proj → QK-norm → dual-theta RoPE → GQA attention (local/global sliding-window
  + causal via Gemma3AttentionLayout) → o_proj → post-attention RMSNorm → +residual → pre-ff RMSNorm →
  GeGLU MLP → post-ff RMSNorm → +residual`. `Gemma3WarpLayerWeights` is the WARP-side per-layer holder
  (`from(Gemma3ReferenceWeights.Layer)`); a new `Gemma3ReferenceForwardPass.runLayer(state, layer)`
  exposes the single-layer oracle. Validated on the real GPU vs the reference for a tiny local layer
  (biting window), a tiny full layer, a 4-heads/2-kv GQA layer, and the **real head_dim=256 geometry**
  (hidden=640, inter=2048) both full and local. Tolerance **abs 2e-3 + rel 2e-3** — looser than the
  element-wise kernels because a layer chains many float kernels against a reference that accumulates
  RMSNorm/RoPE/softmax in double; small realistic weight magnitudes (the normed-input regime). Still a
  building-block composition (CPU readback between kernels), not the fused single-submission pipeline.

- **GEMMA-WARP-9 full prefill — done, real-model GPU-validated.** `Gemma3WarpEmbedding` (lookup
  ×sqrt(hidden), reads only the prompt rows), `Gemma3WarpLmHead` (tied to `embed_tokens`, one GPU upload;
  **heap-light FP32 ByteBuffer** seam + a `float[]` convenience), `Gemma3WarpWeights` (whole-model
  weights, float[] or direct-ByteBuffer embedding), `Gemma3WarpKernels` (shared stateless kernels built
  once across all layers), and `Gemma3WarpForwardPass` (`token ids -> embedding -> 18 layers -> final
  RMSNorm -> tied LM head -> logits`; LM head built lazily so 9a needs no 256k matrix). Validated on the
  real device: synthetic full-prefill exact parity (incl. tied LM head + the heap-light ByteBuffer path
  == float[] path), real-model first-4-layer element parity on BF16 weights, and the **full 18-layer
  real prefill top-1 == 9079 (" Paris")** — matching transformers/the reference. Tolerance abs 3e-3 +
  rel 3e-3 for the synthetic hidden/logits; **top-1 is the asserted metric for the real model** (no
  full 256k-logit identity claim).
  - **Two real-model issues fixed here (both float-vs-double, not logic):**
    1. **GELU-tanh NaN.** The HLSL `tanh` intrinsic (exp-based) overflows float to `inf/inf = NaN` once
       its argument exceeds ~88; Gemma's large real activations (e.g. gate ~= 13 -> arg ~= 88) hit this,
       while synthetic small values never did and the reference uses double `Math.tanh`. Fixed by clamping
       the GELU-tanh argument to +/-20 (tanh saturates to +/-1 there to float precision — mathematically
       exact, overflow-safe). The WARP-6 activation tolerances are unaffected (their args stay < 20).
    2. Shared stateless kernels (`Gemma3WarpKernels`) instead of ~7 per layer — bounded resident
       kernel/PSO/command-list count at 18-layer scale.
  - **Heap note:** the WARP LM head/embedding uses a direct FP32 ByteBuffer decoded from the SafeTensors
    payload (no 164M-weight host `float[]`); the layer projection weights are still host `float[]` (the
    ByteBuffer projection seam exists and can be wired in the product slice). The reference path keeps
    its FP32 `float[]` embedding (parity only).
  - **For GEMMA-WARP-10 (decode session) still missing:** a KV cache (append per step; local layers only
    need a windowed cache), a single-token decode step reusing cached k/v, the generate/streaming loop +
    EOS, and (perf) a fused single-submission pipeline instead of per-call upload/readback.

- **GEMMA-WARP-10a KV cache + single-token decode — done, real-model GPU-validated.** `Gemma3WarpKvCache`
  (per-layer flat `[position][kvDim]` k/v, grows on append; stores the **full** history and applies the
  local/global sliding window at *read* time via `Gemma3AttentionLayout.firstValidKey` — windowed
  eviction is a later perf concern), `Gemma3WarpLayer.decodeStep` (single new token: rmsnorm → q/k/v →
  QK-norm → RoPE at `pos` → append to cache → attention over the visible cached range → o_proj → residual
  → GeGLU MLP → residual, reusing the shared kernels + the layer's projections), and
  `Gemma3WarpDecodeSession` (`prefill` then `decodeNext`; **prefill and decode share the one per-token
  path**, so token `t` attends to `[firstValid(t), t]` exactly as a full-sequence causal pass — prefill's
  last-token logits equal `Gemma3WarpForwardPass`). Tied LM head lazy; reuses `Gemma3WarpKernels`.
  Validated on the real device: synthetic prefill+decodeNext exact parity vs the CPU reference for a
  full-history config **and** a small-sliding-window config (the local window bites as the cache grows),
  cache length grows correctly, and on the real Gemma 3 270M **prefill top-1 = 9079 (" Paris")** and
  **decodeNext(9079) top-1 = 236761**, matching the full-recompute WARP forward pass (the cache path ==
  re-running the whole sequence). Tolerance abs 3e-3 + rel 3e-3 (synthetic logits); top-1 is the asserted
  real-model metric.
  - **For GEMMA-WARP-10b (full generation) still missing:** the multi-step greedy generate loop driving
    `decodeNext` repeatedly, EOS/end-of-turn stopping + the `Gemma3Tokenizer` decode of the produced ids,
    an `IntConsumer`/streaming callback, and (perf, separate) a fused single-submission pipeline +
    windowed-eviction KV cache + the ByteBuffer projection-weight path. No workbench wiring / runtime
    default switch yet.

- **GEMMA-WARP-10b greedy generation — done, real-model GPU-validated.** `Gemma3WarpGenerator`
  (prefill → select → decodeNext → repeat) over the decode session, with `Gemma3TokenSelector` (greedy =
  argmax, sampling-ready seam), `Gemma3StopTokenPolicy` (stop-id set; `ofEos` / `ofEosAndEndOfTurn` —
  Gemma instruct ends a turn with `<end_of_turn>`, not always `<eos>`), `Gemma3GenerationRequest`
  (promptIds + maxNewTokens) and `Gemma3GenerationResult` (`generatedTokenIds`/`fullTokenIds` both
  stop-token-free, `FinishReason` STOP_TOKEN/MAX_TOKENS, prompt/output counts). The stop token ends
  generation, is excluded from the visible output, and is **not** streamed — the `IntConsumer` receives
  exactly `generatedTokenIds`. Validated on the real device: synthetic multi-step greedy equals a manual
  greedy loop over the CPU reference; max-tokens and stop-token (stop on the first token → empty output,
  not streamed) contracts hold; streaming == visible result; and the real Gemma 3 270M generates
  **"The capital of France is" → ids [9079, 236761, 108] = " Paris."** (first token " Paris", MAX_TOKENS
  at 3, streaming consistent) — no text-quality claim beyond the first token + stable execution.
  - **For GEMMA-WARP-11 (workbench) still missing:** a runtime entry point that tokenizes via
    `Gemma3Tokenizer`/`Gemma3ChatTemplate`, runs `Gemma3WarpGenerator`, and decodes to text behind the
    workbench's model/runtime descriptor + a `-Dgemma.runtime=native-warp` flag (external-python stays
    default until proven); plus the heap-light product weight load (ByteBuffer projections, wdmlpack
    payload) and the perf items above. No runtime default switch / downloader / .wdmlpack change here.

- **GEMMA-WARP-11 workbench native flag — done.** `Gemma3RuntimeMode` (`-Dgemma.runtime`: `external`
  default / `native-warp`), `Gemma3NativeWarpRuntime` (tokenizer.json → `Gemma3Tokenizer`/optional
  `Gemma3ChatTemplate` → weights from the compiled `model_gemma3.wdmlpack` via `Gemma3RuntimePackage` →
  `Gemma3WarpGenerator` → detokenized text). `SummarizerPanel.runGemma3Generation` branches on the mode:
  default keeps the **unchanged** external Python path; `native-warp` runs the native runner and logs
  `Runtime mode: native-warp-experimental`, model id/dir, `Backend: WARP`, `Package: model_gemma3.wdmlpack`,
  prompt/output tokens, finish reason. **Missing package with an explicit native flag fails clearly**
  (`"Gemma native WARP requires a compiled .wdmlpack package (...). Use Download/Convert first or run the
  Gemma compiler."`) and does **not** fall back to Python; likewise a missing device fails clearly.
  Validated: runtime-mode parsing (default external, native-warp on flag), the missing-package message is
  actionable, and a real GPU smoke that compiles a `.wdmlpack` and generates **"The capital of France is"
  → " Paris."** through the native runner (`Gemma3NativeWarpRuntimeTest`). external default unchanged,
  not switched. Weights still load through the package's `float[]` reference path (heap) — correct, not
  yet heap-light; the ByteBuffer/heap-light product load + perf remain GEMMA-WARP-12.
  - **Perf note (rough, no optimization claim):** the per-call upload/readback building-block path makes
    the native runner clearly slower than a fused pipeline would be; the gated tests run in the tens of
    seconds (model load + prefill + a few decode steps on the WARP software rasterizer). Real numbers and
    any speedup are GEMMA-WARP-12.
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

- **GEMMA-WARP-12 perf — done (measured, no optimization).** Real numbers in
  `docs/gemma3-warp-runtime-performance.md` (`Gemma3WarpPerformanceProbeTest`, gated). On this WARP/CPU
  host: native-warp decode ≈ **0.9–1.15 tok/s** vs external ≈ **8.8–10.7 tok/s** (~10× slower); native
  prefill is token-by-token and scales badly (~25 s for a 20-token prompt); native heap ≈ **1.2 GB**
  (`float[]` reference weights). Correct (`" Paris."`) but **verdict WAIT** — usable experimentally behind
  the flag, not the sensible default until the fused/batched pipeline + heap-light weight load land.
  Bottlenecks + optimization order are in the perf doc.

- **GEMMA-WARP-13a heap-light product weight load — done.** The native product path no longer uses the
  `float[]` reference weights. `Gemma3RuntimePackage.loadWarpWeightsHeapLight()` mmaps the SafeTensors
  payload and decodes each layer projection (q/k/v/o, gate/up/down) and the tied embedding/LM head into
  **direct FP32 ByteBuffers** (off-heap) via `Gemma3WeightBufferView`; norm vectors stay `float[]`
  (small). `Gemma3WarpLayerWeights` now carries the seven projections as `WarpWeightSource`s
  (ByteBuffer-or-float[], decision in `WarpDenseProjection.fromWeightSource`); `Gemma3WarpMlp`/`Gemma3WarpLayer`
  build from sources — the `float[]` test/reference path is numerically unchanged. `Gemma3NativeWarpRuntime`
  uses the heap-light loader (so the tied embedding/LM head go through `Gemma3WarpLmHead.fromFp32ByteBuffer`,
  not duplicated host-side). **Heap before/after load delta = 0 MB** on the real model (vs ~1199 MB for the
  `float[]` reference path in WARP-12); the ~1 GB of weights now lives off-heap. Validated: synthetic
  ByteBuffer-projection logits == float[] logits (bit-for-bit); real heap-light load yields
  ByteBuffer-backed weights and still prefills top-1 9079 (" Paris"); existing WARP suites unchanged.
  The heavy perf probe is now opt-in (`-Dgemma.perf.probe=true`) so the default suite doesn't OOM the WARP
  device with the added heap-light real-model test. Reference path stays for parity/tests; perf
  (fused/batched pipeline) is the remaining WAIT item before native-warp could become default.

- **GEMMA-WORKBENCH-CONVERT-1 UI Convert — done.** Gemma's artifact lifecycle was `Gemma3DownloadLifecycle`
  (download/probe-only: `hasCompiler=false`, `convert` returns failed) → the Workbench Convert button
  stayed disabled, so `-Dgemma.runtime=native-warp` couldn't get its package from the UI. New
  `Gemma3PackageLifecycle` (mirrors `T5PackageLifecycle`): `hasCompiler=true`, targets exactly
  `model_gemma3.wdmlpack` (`Gemma3WdmlPackCompiler.DEFAULT_OUTPUT_NAME` — the file
  `Gemma3NativeWarpRuntime` loads), `inspect` reports raw (config + safetensors) + package state via
  `Gemma3RuntimePackage`, `convert` calls `Gemma3WdmlPackCompiler.compile(dir, model_gemma3.wdmlpack, force)`.
  Wired into both product sites (`DownloadPanel` rows + `lifecycleSupplier`, and
  `DefaultModelArtifactService.createDefault`). The runtime's missing-package message now matches the UI
  ("Open the Download tab, select google/gemma-3-270m-it, then run Convert."). `Gemma3DownloadLifecycle`
  is retained only for its existing unit test. Validated: device-free lifecycle tests (raw-complete →
  Convert, empty → Inspect, corrupt package → Repair, exact-name `existingPackage`) and a CPU
  real-convert test (hard-links the model into a temp dir, converts, asserts the package is
  runtime-loadable and the runtime finds it, lifecycle then READY). No native-runtime path change beyond
  the message; no format change; T5/SmolLM2/Qwen lifecycles untouched.
