# Phi-3 wdmlpack compiler — requirements audit + foundation

> **Status (PHI3-WDMLPACK-COMPILER-1/2): heap-safe compiler + package built; Workbench stays PLANNED.** The compiler
> (`Phi3WdmlPackCompiler`), role catalog (`Phi3WdmlPackRoles`), package loader (`Phi3RuntimePackage` +
> `Phi3Weights.ofRecords`), an ONNX import boundary (`Phi3OnnxModelSource`) and a heap-light layout planner
> (`Phi3Weights.planLayout` + `Phi3WeightLayout`) exist. The serialization core is proven by a **non-gated synthetic
> round-trip** (byte-exact: INT4 triplets + fp32 vectors + config). **COMPILER-2 made the real compile heap-safe:**
> `Phi3WdmlPackCompiler.compile` now **streams** every tensor straight from the mmap'd `model.onnx.data` (copy raw
> INT4 / convert fp16→fp32 in bounded buffers) via `planLayout`, never materializing the full ~2.4 GB model. The
> gated real Phi-3-mini compile (`-Dphi3.compile.realModel=true`) now **runs within the standard 2 GB test heap** and
> produces a real `model_phi3.wdmlpack` — **711 tensors, 32 layers, ~2.39 GB**.
>
> **>2 GB reader fixed (WDMLPACK-LARGE-READER-1).** The shared reader now handles packages larger than 2 GB without a
> format change: `WdmlPackWriter.readManifest` reads only the manifest region positionally (no whole-file mmap), and
> `WdmlPackReader.mapPayloadTensors` maps each tensor's payload region individually with a long file offset (each
> tensor is < 2 GB). The real ~2.39 GB Phi-3-mini package now **reloads** — `Phi3RuntimePackage.open` +
> `runtimeTensorCatalog()` validate config + all 711 role tensors (gated test). Small Qwen/T5/SmolLM2 packages are
> unchanged (regression green). The remaining step before runnable is the full in-memory `weights()` reconstruction
> footprint (~3 GB) — a runtime-memory concern for the Workbench wiring, not the reader. **No Workbench
> lifecycle/gate/download wiring and no status flip**; Phi-3 remains PLANNED / gate-blocked. Next:
> `PHI3-WORKBENCH-RUNNABLE-1`.

---

# Phi-3 wdmlpack compiler — requirements audit (PHI-WDMLPACK-COMPILER-AUDIT-1)

Spec-only. No compiler, no runtime change, no kernels, no product unlock. This documents what a minimal
`Phi-3-mini` wdmlpack compiler would need so Phi-3 can later become honestly runnable in the Workbench — exactly
like Gemma/Qwen/SmolLM2/T5 — and bounds the follow-up work.

## Where the Phi code lives today

Library/sidecar runtime (`directml-inference`), **not** wired into the Workbench product path:
- `inference/Phi3InferenceEngine.java` — engine: loads config/tokenizer/weights, runs `Phi3Runtime`. Backends
  `cpu` / `directml` / `auto` (projections on GPU via `Phi3GpuKernels`/`Phi3GpuPipeline`, attention/norms on CPU).
  **No Python, no ONNX Runtime** — it reads ONNX only as a weight container.
- `inference/Phi3Summarizer.java` — `Summarizer` wrapper over the engine (sidecar `summarize` JSON-RPC).
- `inference/phi3/Phi3Weights.java` — **the load-bearing piece.** Parses the ONNX graph topology (MatMulNBits
  nodes + output-name tracing like `/model/layers.N/input_layernorm/Mul_1_output_0`) and a custom external-data
  offset parser to extract INT4 (Q4G128: qWeight+scales+zeroPoints, block 128) projections + fp16 norms/embedding/
  cos-sin into in-memory `QuantizedWeight` / `LayerWeights` records.
- `inference/phi3/{Phi3Config,Phi3Runtime,Phi3GpuKernels,Phi3GpuPipeline,Phi3Tokenizer,SimdOps}.java`,
  `windows/Phi3ComputeShaders.java`, `inference/prompt/Phi3PromptStrategy.java`, `inference/bench/Phi3Benchmark.java`.
- Tests: `Phi3InferenceEngineModelDirTest`, `Phi3Benchmark` (no real-model gated cert).

## Artifacts Phi-3-mini uses

Source (HF `microsoft/Phi-3-mini-4k-instruct-onnx`, DirectML INT4 build, ~2.3 GB), validated by
`Phi3InferenceEngine.describeMissingModelFile`:
- `config.json` (`Phi3Config`), `tokenizer.json` (`Phi3Tokenizer`, SentencePiece),
- `model.onnx` (graph/weight metadata), `model.onnx.data` (~2.1 GB external INT4 + fp16 weights).

Weight formats inside: INT4 MatMulNBits (`Q4G128`) for q/k/v/o/gate_up/down + lm_head; fp16 for norms, embedding,
cos/sin cache, activation scales.

## What the Workbench package path expects (the gap)

The Workbench routes every runtime through `WorkbenchArtifactGate.requireExecutablePackage(family, dir)` →
`DefaultModelArtifactService` → a per-family `ModelPackageLifecycle`. Executable families (Gemma/Qwen/SmolLM2/T5)
have a real `*PackageLifecycle` (`hasCompiler()=true`) that compiles a `.wdmlpack` and a `*RuntimePackage` that
loads it. **Phi-3 is wired as `CompilerMissingLifecycle`** (no compiler) → `inspect()` is always
`PACKAGE_COMPILER_MISSING`/`executable=false` → the gate always throws. So the Workbench has no executable Phi path;
the homogeneous lifecycle deliberately refuses to run families from raw weights.

The wdmlpack pipeline a Phi compiler must join (shared `inference/model/`): `ModelSource` → `SourceTensorCatalog`
→ tensor-role mapping → `WdmlPackWriter` → `RuntimePackage.open` (`WdmlPackReader` → `RuntimeTensor` catalog) →
runtime. `RuntimeTensor` carries a generic `dataType` + raw bytes, so INT4 triplets (q/scale/zp) round-trip as
raw tensors.

## Reusable vs. new

**Reusable as-is (no change):**
- The entire Phi **runtime**: `Phi3Runtime`, `Phi3GpuKernels`, `Phi3GpuPipeline`, `Phi3ComputeShaders`,
  `Phi3Tokenizer`, `Phi3Config`, `Phi3PromptStrategy`. They operate on in-memory `Phi3Weights` records, independent
  of where those bytes came from.
- The shared wdmlpack/format infra: `WdmlPackWriter/Reader`, `RuntimeTensor(Catalog)`, `TensorCatalog`,
  `RuntimeModelPackage`, `RuntimeLoadability`.
- **Strong precedent — Qwen already does ONNX(INT4)→wdmlpack:** `QwenOnnxModelSource` (`OnnxModelReader.parse` +
  external-ref parsing + INT4 `SourceTensor`s), `QwenWdmlPackCompiler`, `QwenPackageLifecycle`,
  `QwenWdmlPackModelSource`. This is the closest template; the wdmlpack INT4 format is **already defined and shipping**
  (Qwen `model_q4f16.wdmlpack`), so no new format needs to be invented.
- Phi's ONNX extraction logic in `Phi3Weights` (graph tracing + `parseExternalRefs`) — reusable as the reader inside
  a Phi ONNX `ModelSource`.

**New work (Phi-specific), mirroring Qwen/SmolLM2:**
1. **Phi ONNX `ModelSource`** — wrap the `Phi3Weights` graph-tracing/external-data extraction behind the
   `ModelSource`/`SourceTensorCatalog` boundary (or factor the shared ONNX-external-ref parser out of
   `Qwen2Weights`/`Phi3Weights` to avoid a third copy).
2. **Phi tensor-role mapping** — stable role names (per-layer q/k/v/o/gate_up/down, input/post norms, embed,
   final norm, lm_head, cos/sin, activation scales) ↔ the ONNX graph-traced tensors. The discovery already exists in
   `Phi3Weights.loadLayer`; it must be expressed as a role catalog (à la `SmolLM2TensorNameMapper`/`TensorRole`).
3. **`Phi3WdmlPackCompiler` + `Phi3CompileOptions` + compile tool/report** — serialize the INT4 triplets +
   fp16 tensors into a `model_phi3.wdmlpack` via `WdmlPackWriter` (Qwen compiler as template).
4. **`Phi3RuntimePackage` + manifest/metadata + `Phi3RuntimeLoadability`** — open + validate the package.
5. **Package-backed `Phi3Weights` loader** — reconstruct `Phi3Weights` (`QuantizedWeight`/`LayerWeights`) from the
   package's `RuntimeTensor`s, so `Phi3Runtime` is reused unchanged. (The one real adapter seam.)
6. **`Phi3PackageLifecycle`** (`hasCompiler()=true`) replacing `CompilerMissingLifecycle(PHI3)` in
   `WorkbenchArtifactGate` + `DefaultModelArtifactService` + the `DownloadPanel` Phi row (Convert).
7. **Status flip** — once executable: Phi-3-mini `PLANNED → EXPERIMENTAL`; update `SummarizerPanel` note +
   `WorkbenchModelStatusAuditTest` runnable set + docs.
8. **Tests** — compiler test, runtime-package loadability, wdmlpack-vs-ONNX parity, and a gated real-model cert
   (mirror `T5MixedRuntimeCorrectnessCertTest`).

## Phi-3.5 (boundary only — no work here)

`microsoft/Phi-3.5-mini-instruct-onnx` is the same architecture class (`Phi3Config`, same ONNX MatMulNBits INT4
layout) and would use the **same** compiler/runtime once its ONNX-format weights are published — expected to need no
new mapping, pending verification that its graph node-name patterns match. Stays PLANNED; not part of any build here.

## Decision: B — a real new compiler is needed, but the runtime is largely reusable

Not **A** (thin adapter): there is no Phi ONNX `ModelSource`, no Phi package/loader/lifecycle, and the INT4
graph-traced extraction must be re-expressed as a role catalog + serialized — genuinely new classes (≈ the Qwen set).
Not **C** (re-architecture): the Phi **runtime**, the **wdmlpack INT4 format**, the **ONNX→wdmlpack pattern**
(Qwen), and the ONNX extraction logic all already exist and need no restructuring. The package-backed `Phi3Weights`
loader is the only real new seam in the runtime; everything else is compiler/lifecycle plumbing that closely mirrors
Qwen.

**Workbench status stays PLANNED / gate-blocked** until that compiler exists — unchanged by this slice. Suggested
follow-up sequence: `PHI3-WDMLPACK-COMPILER-1` (ONNX source + role map + compiler + package, with a wdmlpack-vs-ONNX
parity test, runtime still library-only) → `PHI3-WORKBENCH-RUNNABLE-1` (lifecycle/gate/download wiring + status flip
+ gated real-model cert).

## Out of scope (this slice)

No compiler/runtime/kernel code, no `.wdmlpack` format change, no lifecycle/gate/download wiring, no status flip, no
real-model test un-gating. Audit + spec only.
