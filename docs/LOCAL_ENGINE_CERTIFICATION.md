# Local Engine Certification (feature/askai-local-engine-artifacts)

Real-model certification of the neutral public generation runtime
(`com.aresstack.windirectml.inference.api.GenerationRuntime`). Each model is certified only when
the full production chain runs green:

```
download → compile to *.wdmlpack (lifecycle.convert) → package-only directory WITHOUT raw weights
  → GenerationRuntime PACKAGE_ONLY load → inference → unload → reload → inference
  → fail-closed check (missing package ⇒ PACKAGE_MISSING)
```

Driven by the opt-in `PublicApiGenerationSmokeIT` (structural invariants: non-empty text,
generated-token count > 0, streamed tokens, finish reason, no NaN/Infinity). Host: Windows 11,
RTX 5080 (16 GB), Zulu 25, Gradle 9.0.

**Status vocabulary** — `REAL_CERTIFIED` (full package-only chain green on a real model via the
public API) · `PROVEN_RUNTIME_REUSED` (wired to the exact shipping engine path; public-API run not
executed here) · `CONTRACT_TESTED` (code + dispatch tests green; real run not possible here) ·
`EXTERNALLY_BLOCKED` (gated/credential) · `FAILED`. Open findings: see `problems.md`.

Reproduce one model:

```
gradlew :directml-inference:test --tests '*PublicApiGenerationSmokeIT' ^
  -Dsmoke.repo=<hf-repo> -Dsmoke.rawDir=<downloaded-dir> -Dsmoke.backend=<cpu|auto|warp>
```

## Seq2Seq — T5 family (adapter: T5GenerationAdapter → T5InferenceEngine)

| Model | Source | CPU | AUTO (HW) | WARP (SW) | Notes |
|---|---|---|---|---|---|
| google-t5/t5-small | safetensors | ✅ | ✅ | ✅ | full backend matrix |
| google/flan-t5-small | safetensors | ✅ | ✅ | — | backend path shared w/ t5-small |
| Salesforce/codet5-small | torch checkpoint (pytorch_model.bin) | ✅ | — | — | BPE tokenizer (vocab.json+merges.txt) |
| Salesforce/codet5-base-multi-sum | torch checkpoint (892 MB) | ✅ | — | — | BPE tokenizer |

All four T5 models are **REAL_CERTIFIED**. The WARP/AUTO DirectML path is engine-level (identical
across all T5 checkpoints) and is fully proven on t5-small across CPU/AUTO/WARP. Per-variant CPU
certification exercises each model's own
weight-import (safetensors vs. torch checkpoint) and tokenizer. No code fixes were required for the
T5 family beyond keying the prompt strategy off the HF repository id.

## Causal / Chat

The **test-run state** (what a real run proved here) is kept separate from the **catalog status** (the
release gate — whether a host may offer the model as a local recommendation). Only a model with a real
green run is `RUNNABLE`; the one accepted exception is Qwen (see below).

| Model | Family | CPU | AUTO | WARP | Test-run state | Catalog |
|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-Coder-0.5B-Instruct | QWEN | ⟳ | ⟳ | ⟳ | **PROVEN_RUNTIME_REUSED*** | **RUNNABLE** (accepted exception*) |
| HuggingFaceTB/SmolLM2-135M-Instruct | SMOLLM2 | ✅ | ✅ (HW) | removed‡ | **REAL_CERTIFIED** (CPU+AUTO) | **RUNNABLE** |
| HuggingFaceTB/SmolLM2-360M-Instruct | SMOLLM2 | ✅ | ✅ (HW) | removed‡ | **REAL_CERTIFIED** (CPU+AUTO) | **RUNNABLE** |
| microsoft/Phi-3-mini-4k-instruct-onnx | PHI3 | wired | n/a | n/a | **CONTRACT_TESTED** — matrix limited to CPU (P2); real run pending, no weights (P6) | **UNVERIFIED** |
| google/gemma-3-270m-it | GEMMA3 | n/a | ⛔ | ⛔ | **EXTERNALLY_BLOCKED** — adapter + contract test built (P4)† | **UNVERIFIED** |

Matrix note: SmolLM2 catalog backends narrowed to `{AUTO, CPU}` (WARP withheld, P1); Phi-3 narrowed
to `{CPU}` (package-backed GPU not wired, P2). Phi-3 and Gemma keep full descriptors/manifests/backends
but their catalog status is **UNVERIFIED** — they are absent from `LocalModelCatalog.runnable()` and are
not offered as local recommendations until a real green public-API package-only run promotes them.
`EXTERNALLY_BLOCKED`/`CONTRACT_TESTED` describe the test run; the catalog release status is `UNVERIFIED`.

\* **Qwen — RUNNABLE as an explicitly accepted release exception (per instruction).** Qwen is
`PROVEN_RUNTIME_REUSED`, not `REAL_CERTIFIED`: the public-API package-only run was skipped by user
decision. It is nonetheless kept `RUNNABLE` in the catalog as a documented exception (see the release
gate in `RELEASE_0.2.0.md`), because the adapter reuses the exact shipping engine path.
`QwenGenerationAdapter` wraps the same
`QwenInferenceEngine` the shipping workbench (`SummarizerPanel.runQwenGeneration`) uses, on the
identical construction + backend path. One wiring nuance was fixed: the adapter now derives the ONNX
variant name from the catalog package name (`model_q4f16.wdmlpack` → `model_q4f16.onnx`) so the
engine's package resolution matches the catalog (the 3-arg default `model.onnx` would have looked for
`model.wdmlpack`). The public-API package-only run was skipped at the user's request; reproduce with:

```
gradlew :directml-inference:test --tests '*PublicApiGenerationSmokeIT' ^
  -Dsmoke.repo=Qwen/Qwen2.5-Coder-0.5B-Instruct -Dsmoke.rawDir=<dir-with-model_q4f16.onnx+config+tokenizer> ^
  -Dsmoke.backend=cpu
```

‡ **SmolLM2 software-WARP finding (P1).** With the real 135M model, the D3D12 WARP *software*
rasterizer produced empty output through the public API (CPU and AUTO/hardware are fine); the
fixture-based WARP unit tests pass, so this is a real-weights end-to-end gap. **Decision:** WARP is
removed from the SmolLM2 catalog matrix (now `{AUTO, CPU}`) rather than offered while broken, and the
adapter fails closed on an explicit WARP request instead of silently using CPU. Reproduce/fix:
`-Dsmoke.repo=HuggingFaceTB/SmolLM2-135M-Instruct -Dsmoke.backend=warp`.

† **Gemma 3 externally blocked.** `google/gemma-3-270m-it` is a gated Hugging Face repo and no HF
token is present in this environment, so the model cannot be downloaded/certified here. The Gemma
adapter (`Gemma3GenerationAdapter` → `Gemma3NativeWarpRuntime`, WARP/AUTO only, no Python) is built
and covered by a contract test that proves dispatch. Reproduce once a token is available:

```
setx HF_TOKEN <token>   &&  huggingface-cli download google/gemma-3-270m-it --local-dir <dir>
gradlew :directml-inference:test --tests '*PublicApiGenerationSmokeIT' ^
  -Dsmoke.repo=google/gemma-3-270m-it -Dsmoke.rawDir=<dir> -Dsmoke.backend=auto
```

## Reranker (W6)

| Model | Compile | Ranking A>B (CPU) | Ranking A>B (WARP) | Package-only load | Catalog status |
|---|---|---|---|---|---|
| cross-encoder/ms-marco-MiniLM-L12-v2 | ✅ | ✅ | ✅ | ✅ code + CPU (WARP pending) | **UNVERIFIED** |

**P3 is code-fixed** (`CODE_FIXED, REAL_WARP_CERTIFICATION_PENDING`). The reranker load path already read
weights exclusively from `reranker.wdmlpack`; the completeness check now requires that package instead of
the raw `model.safetensors`, so a package-only directory loads (proven device-free by
`RerankerPackageOnlyLoadTest` and with a real CPU rank on the shipped L6 package). L12 ranking is correct
on CPU and WARP through the public `LocalMlRuntime` facade (`PF4J plugin framework` → the PF4J doc
outranks the off-topic doc). L12 stays **UNVERIFIED** in the catalog until the opt-in real package-only
ranking smoke runs green on **CPU and WARP** (still outstanding: L6 package-only WARP, L12 package-only
CPU + WARP — needs the L12 download + a healthy GPU, see P9). Promote L12 to RUNNABLE only then. No
catalog status change was made to L12 in this slice.
