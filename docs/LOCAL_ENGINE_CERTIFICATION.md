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

| Model | Family | CPU | AUTO | WARP | Status |
|---|---|---|---|---|---|
| Qwen/Qwen2.5-Coder-0.5B-Instruct | QWEN | ⟳ | ⟳ | ⟳ | **PROVEN_RUNTIME_REUSED*** |
| HuggingFaceTB/SmolLM2-135M-Instruct | SMOLLM2 | ✅ | ✅ (HW) | removed‡ | **REAL_CERTIFIED** (CPU+AUTO) |
| HuggingFaceTB/SmolLM2-360M-Instruct | SMOLLM2 | ✅ | ✅ (HW) | removed‡ | **REAL_CERTIFIED** (CPU+AUTO) |
| microsoft/Phi-3-mini-4k-instruct-onnx | PHI3 | wired | n/a | n/a | **CONTRACT_TESTED** — matrix limited to CPU (P2); real run pending, no weights (P6) |
| google/gemma-3-270m-it | GEMMA3 | n/a | ⛔ | ⛔ | **EXTERNALLY_BLOCKED** — adapter + contract test built (P4)† |

Matrix note: SmolLM2 catalog backends narrowed to `{AUTO, CPU}` (WARP withheld, P1); Phi-3 narrowed
to `{CPU}` (package-backed GPU not wired, P2).

\* **Qwen — certified by workbench reuse (per instruction).** `QwenGenerationAdapter` wraps the same
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
| cross-encoder/ms-marco-MiniLM-L12-v2 | ✅ | ✅ | ✅ | ❌ (P3) | **UNVERIFIED** |

L12 ranking is correct on CPU and WARP through the public `LocalMlRuntime` facade (`PF4J plugin
framework` → the PF4J doc outranks the off-topic doc). It stays **UNVERIFIED** because the mandatory
package-only load is not satisfiable: the reranker load path requires `model.safetensors` present
even though weights come from `reranker.wdmlpack` (P3 in problems.md). Promote to RUNNABLE only after
the completeness check is relaxed and the package-only test passes. No catalog change was made to L12.
