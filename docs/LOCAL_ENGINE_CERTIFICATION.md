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

The WARP/AUTO DirectML path is engine-level (identical across all T5 checkpoints) and is fully
proven on t5-small across CPU/AUTO/WARP. Per-variant CPU certification exercises each model's own
weight-import (safetensors vs. torch checkpoint) and tokenizer. No code fixes were required for the
T5 family beyond keying the prompt strategy off the HF repository id.

## Causal / Chat

| Model | Family | CPU | AUTO | WARP | Status |
|---|---|---|---|---|---|
| Qwen/Qwen2.5-Coder-0.5B-Instruct | QWEN | — | — | — | pending |
| HuggingFaceTB/SmolLM2-135M-Instruct | SMOLLM2 | — | — | — | pending (adapter W3.2) |
| HuggingFaceTB/SmolLM2-360M-Instruct | SMOLLM2 | — | — | — | pending (adapter W3.2) |
| microsoft/Phi-3-mini-4k-instruct-onnx | PHI3 | — | — | n/a | pending |
| google/gemma-3-270m-it | GEMMA3 | n/a | — | — | pending (gated; needs HF token) |

## Reranker (W6)

| Model | CPU | DirectML/WARP | Ranking (A>B) | Catalog status |
|---|---|---|---|---|
| cross-encoder/ms-marco-MiniLM-L12-v2 | — | — | — | UNVERIFIED (pending W6) |
