# Open Problems / Findings — feature/askai-local-engine-artifacts

Findings surfaced while certifying the neutral generation runtime and reranker with real models.
Each has a precise cause, current handling, and a fix path. None blocks the generation runtime for
its certified backends; several intentionally narrow a catalog matrix to stay truthful.

Status vocabulary used in the certification log and final report:

```
REAL_CERTIFIED       full package-only chain run green on a real model via the public API
PROVEN_RUNTIME_REUSED wired to the exact shipping engine path; public-API run not executed here
CONTRACT_TESTED      code + dispatch/contract tests green; real model run not possible here
EXTERNALLY_BLOCKED   cannot run in this environment (gated download / missing credential)
FAILED               real run attempted and failed
```

---

## P1 — SmolLM2 software-WARP produces empty output (real weights)

- **Symptom:** With the real SmolLM2-135M model, loading through the D3D12 WARP *software* rasterizer
  and generating via the public API returns empty text. CPU (reference) and AUTO (RTX 5080 hardware)
  are green.
- **Not caught by:** the existing SmolLM2 WARP unit tests (17, fixture-based) all pass — this is a
  real-weights end-to-end gap.
- **Handling:** WARP removed from the SmolLM2 catalog matrix (now `{AUTO, CPU}`). The adapter never
  silently falls back for an explicit backend: explicit WARP fails closed (`UNSUPPORTED_BACKEND`),
  AUTO may resolve to CPU and reports the **actual** backend used.
- **Fix path:** debug the SmolLM2 WARP generation kernel with real weights (numerics/detokenization),
  then re-run `-Dsmoke.repo=HuggingFaceTB/SmolLM2-135M-Instruct -Dsmoke.backend=warp` and restore WARP
  to the matrix.

## P2 — Phi-3 package-backed runtime is CPU-only today

- **Cause:** the package-backed Phi-3 path (`Phi3RuntimePackage` + `Phi3Runtime`) has only a CPU
  compute path. The DirectML kernels (`Phi3GpuKernels`/`Phi3GpuPipeline`) live in the raw-ONNX
  `Phi3InferenceEngine`, which loads `model.onnx`+`model.onnx.data` (not package-only) and is not
  wired to the package weights.
- **Handling:** Phi-3 catalog matrix limited to `{CPU}`. Non-implemented backends return
  `UNSUPPORTED_BACKEND` (not an init failure). Phi-3 real run is also pending — no Phi-3 weights are
  present locally (the repo scaffold has only a `model.onnx.data.url` stub).
- **Fix path:** feed `Phi3RuntimePackage.weights()` into `Phi3GpuKernels`/`Phi3GpuPipeline`, add a
  package-backed DIRECTML/AUTO path, real-test it, then restore DIRECTML/AUTO to the matrix.

## P3 — Reranker load is not strictly package-only

- **Cause:** `BertCrossEncoderRerankers.REQUIRED_FILES` = `{model.safetensors, tokenizer.json,
  config.json}` — the reranker load validates that `model.safetensors` is **present** even though the
  weights are read from `reranker.wdmlpack`. A package-only directory (no raw weights) therefore
  fails with "Reranker model directory is incomplete".
- **Impact:** cross-encoder/ms-marco-MiniLM-L12-v2 **ranking is correct** (compile + load + score
  A>B green on CPU and WARP), but the mandatory package-only load is not satisfied, so **L12 stays
  UNVERIFIED**. The same limitation affects the already-RUNNABLE L6 under a strict package-only bar.
- **Fix path:** relax the reranker completeness check to accept `reranker.wdmlpack` in lieu of
  `model.safetensors`, and confirm `DirectMlReranker` reads weights from the package (not the loose
  safetensors); then re-run the L12 package-only test and promote L12 to RUNNABLE.

## P4 — Gemma 3 real run externally blocked

- **Cause:** `google/gemma-3-270m-it` is a gated HF repo and no HF token is present in this
  environment.
- **Handling:** the Gemma adapter (`Gemma3GenerationAdapter` → `Gemma3NativeWarpRuntime`, WARP/AUTO
  only, no Python) is built and covered by a dispatch contract test. Status: `EXTERNALLY_BLOCKED`.
- **Retry:** set `HF_TOKEN`, download the model, then
  `-Dsmoke.repo=google/gemma-3-270m-it -Dsmoke.rawDir=<dir> -Dsmoke.backend=auto`.

## P5 — Qwen public-API package-only run not executed (by instruction)

- The Qwen adapter reuses the exact shipping `QwenInferenceEngine` path; per instruction the public
  package-only run was not executed. Status: `PROVEN_RUNTIME_REUSED`. A wiring bug was found and fixed
  in the process (the adapter now derives `model_q4f16.onnx` from the catalog package name so the
  engine resolves `model_q4f16.wdmlpack`). Reproduce command in the certification log.

## P7 — Neutral API has no PromptTask; workbench task selector degraded on the shared path

- The public `GenerationRequest` models `systemPrompt` + `userPrompt` only (AskAI maps its own
  chat/generate). The workbench's `PromptTask` selector (summarize/translate/explain) is therefore
  not applied when routing through `WorkbenchGenerationService`; the family chat template still wraps
  the user text. Acceptable for AskAI; a minor workbench UX regression.
- **Fix path:** either map each `PromptTask` to a short instruction the panel prepends as the system
  prompt, or add a neutral task/instruction concept to the API. Tracked as a manual-GUI remainder.

## P8 — Dormant per-family methods + Python Gemma runner remain in the workbench

- The workbench's active generation dispatch now goes through the shared runtime (W4), but the old
  per-family methods (`runQwenGeneration`/`runT5Generation`/`runSmolLm2Generation`/`runGemma3*`/
  `runPhi3Summarizer`) and `Gemma3ExternalRuntimeRunner` (+ `gemma3_generate.py`) are left in place,
  dormant, to avoid a blind mass deletion in a GUI file that cannot be run headlessly here.
- **Fix path:** delete the dormant methods and the Python runner under interactive GUI verification
  (they are no longer invoked). The published runtime (`directml-inference`) is already Python-free
  (enforced by `RuntimeArchitectureTest`); this is workbench-only cleanup.

## P6 — Phi-3 real run pending (no local weights)

- The package-backed CPU Phi-3 path is wired and compiles, but no Phi-3 weights are present locally
  to run the real package-only chain. Status: `CONTRACT_TESTED`. Reproduce once weights are present
  with `-Dsmoke.repo=microsoft/Phi-3-mini-4k-instruct-onnx -Dsmoke.backend=cpu`.
