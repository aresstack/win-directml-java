# Workbench model runtime status (WORKBENCH-MODEL-STATUS-1)

Audit of every generation model family in the Workbench: is it selectable, is it executable, which backend
path runs for WARP / AUTO / CPU, and is any stale `planned` / `probe` / `experimental` /
"Runtime integration is planned" wording or silent Python fallback left over. Gemma is the reference
(WARP/AUTO native, CPU legacy external, no visible research mode).

Audit-only slice: no runtime optimization, no migration. The state below is what the code does today; the
prior Gemma slices (PRODUCT-1/2 + closeout) and the homogeneous-lifecycle work already removed the
user-visible residue, so this slice confirms it and locks it in with `WorkbenchModelStatusAuditTest`.

## Status table

| Model family | Dropdown | WARP | AUTO | CPU | Registry status | Executable | Remark |
|--------------|----------|------|------|-----|-----------------|------------|--------|
| **Gemma 3 270M-it** | yes | native DirectML (WARP software adapter) | native DirectML (first hardware adapter, optional) | legacy external Python | EXPERIMENTAL (internal, not shown) | **yes — product** | Closed out. Download tab = `Gemma3PackageLifecycle` (Convert/READY). No visible research label (runtime label `native-warp`). |
| Gemma 3 270M (base) | yes | native (no product package) | native | legacy external Python | PLANNED | no | Base checkpoint; selecting it without a package → clear Download→Convert message (no silent Python). |
| **Qwen2.5-Coder-0.5B-it** | yes | native DirectML INT4 (`model_q4f16.wdmlpack`) | native DirectML INT4 | engine backend | EXPERIMENTAL | **yes** | Runnable by status (WORKBENCH-MODEL-STATUS-2) — the PLANNED-guard `qwenTestModel` exemption was removed; `isQwenTestModel` now only routes to `QwenInferenceEngine` (like the other families' routing). No Python. |
| Qwen2.5-Coder 1.5B / 3B | yes | — | — | — | PLANNED | no | "selectable but not executable yet" (honest guard message). |
| **SmolLM2-135M / 360M** | yes | native dense projections on the D3D12 software rasterizer (CPU) | hardware GPU when present, else Java reference | native | EXPERIMENTAL | **yes — from `.wdmlpack`** | No Python. Requires a prebuilt package (Download→Convert). |
| **T5 / Flan-T5 / CodeT5** | yes | native from `.wdmlpack` | native | native | EXPERIMENTAL | **yes — from `.wdmlpack`** | Seq2seq. No Python. `T5InferenceEngine`. |
| **Phi-3 mini (onnx)** | yes | `Phi3Summarizer`/`Phi3InferenceEngine` (backend forwarded) | same | same | EXPERIMENTAL | yes (runtime present) | Runs via the Phi-3 engine; the panel notes "no direct ONNX execution / from a wdmlpack". Exact engine internals not re-verified in this Gemma-scoped audit. No Python. |
| Phi-3.5 mini (onnx) | yes | — | — | — | PLANNED | no | "selectable but not executable yet". |

## Findings

- **No user-visible stale wording.** The dropdown lists model ids only (no status tooltip). The only visible
  research term had been the Gemma runtime label `native-warp-experimental` → already fixed to `native-warp`
  (PRODUCT-2) and the Gemma "probe" notes → "legacy external Python" (closeout). No `Runtime integration is
  planned` string exists anywhere in the repo (verified); locked by `WorkbenchModelStatusAuditTest`.
- **Executable models are not blocked.** The Summarizer's PLANNED guard only fires for genuinely
  not-executable PLANNED models (Phi-3.5, Qwen 1.5B/3B, Gemma base); Gemma/SmolLM2/T5/Phi-3 and Qwen 0.5B
  are EXPERIMENTAL (runnable). Qwen 0.5B is runnable by status — the `qwenTestModel` PLANNED-guard exemption
  was removed in WORKBENCH-MODEL-STATUS-2 (the predicate now only routes to `QwenInferenceEngine`). Locked by
  the audit test.
- **Not-executable models are clearly marked**, both in the Summarizer ("selectable but not executable yet.
  Status: planned…") and in the Download tab (`package compiler not implemented (downloadable, not
  executable)` for families without a compiler).
- **Python only where allowed.** The single Python path is the Gemma `Backend = CPU` legacy external
  runner. WARP/AUTO never use Python/Transformers/ONNX-Runtime; no silent Python fallback (a missing Gemma
  package fails with a clear Download→Convert message).
- **Download-tab status is correct for Gemma.** `DefaultModelArtifactService` wires `Gemma3PackageLifecycle`
  (`hasCompiler()=true` → Convert/READY). The old `Gemma3DownloadLifecycle` (download/probe-only) and the
  `ModelArtifactRow.gemmaDownloadOnlyStatusText` branch are now unreachable for Gemma (dead remnants, not
  user-visible); left in place to keep this audit slice minimal — candidates for a later cleanup.

## Gemma reference (unchanged)

Gemma stays the reference product path and is untouched by this slice: `Backend = WARP` → native
Java/DirectML on the WARP software adapter (no Python); `Backend = AUTO` → native on the first hardware
adapter (optional, clear message when none); `Backend = CPU` → legacy external Python only. Paris smoke
(" Paris", token 9079) green on WARP and HARDWARE (gated tests). See `gemma3-warp-runtime-plan.md`
(Status: COMPLETE).
