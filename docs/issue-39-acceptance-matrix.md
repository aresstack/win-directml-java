# Issue #39 acceptance matrix

Tracking issue: `#39`  
PR linkage rule for partial work: **Part of #39** / **Refs #39** (do not close #39 early).

## Current acceptance matrix

| Criterion | Status | Code evidence | Test evidence | Doc evidence | Notes |
|---|---|---|---|---|---|
| Shared company-model registry exists in Java-8 module | ✅ | `directml-config/.../EmbeddingModelRegistry.java` | `EmbeddingModelRegistryTest` | `SUPPORTED_MODELS.md` §1.1 | Shared by Java-21 sidecar and Java-8 workbench/client modules. |
| Sidecar rejects non-embedding IDs with explicit messages | ✅ | `DirectMlPhi3Sidecar.embedFamily(...)` | `EmbedFamilyParseTest` | `SUPPORTED_MODELS.md` §1.1 | Decoder/summarizer IDs are blocked. |
| Planned embedding IDs without runtime implementation are blocked | ✅ | `embedFamily(...)` + `unimplementedEmbeddingErrorMessage(...)` | `EmbedFamilyParseTest` | `SUPPORTED_MODELS.md` §1.1 | Covers Jina + multilingual-E5 planned IDs. |
| Workbench embed selector excludes non-embedding IDs | ✅ | `EmbedModelOptions.embeddingOptions()` | `EmbedModelOptionsTest` | `WORKBENCH.md` | Decoder/summarizer hidden in UI. |
| Workbench embed selector excludes planned/unimplemented embedding IDs | ✅ | `EmbedModelOptions.embeddingOptions()` (`embedFamily != null`) | `EmbedModelOptionsTest.plannedEmbeddingModelsWithoutRuntimeSupportAreNotSelectable` | `SUPPORTED_MODELS.md` §1.1 | Prevents planned models from appearing silently supported. |
| E5 base-sts-en-de shipped evidence package | 🟡 | `E5Encoders`, sidecar `-Dembed.model=e5` path | `E5RealModelReferenceTest`, `DirectMlEmbedBatchParityTest` | `README.md`, `PROTOCOL.md`, `SUPPORTED_MODELS.md` | Windows real-model run remains release-gate evidence. |
| MiniLM shipped evidence package | 🟡 | `CpuMiniLmEncoder`, `DirectMlMiniLmEncoder` | `EmbeddingReferenceTest`, `DirectMlMiniLmEmbeddingReferenceTest` | `README.md`, `SUPPORTED_MODELS.md` | Windows DirectML evidence remains release-gate evidence. |
| Jina feasibility and implementation delta documented | ✅ | classification in registry (`planned`) | n/a (analysis-only) | `docs/model-analysis-jina-embeddings-v2-base-de.md` | Do not mark shipped before real-model tests. |
| multilingual-E5-instruct feasibility and implementation delta documented | ✅ | classification in registry (`planned`) | n/a (analysis-only) | `docs/model-analysis-multilingual-e5-large-instruct.md` | Do not mark shipped before SentencePiece/XLM-R work + real-model tests. |
| Benchmark docs for supported embedding models | 🟡 | `EmbedBatchBenchmark` harness | n/a (runtime measurements) | `directml-encoder/BENCHMARK.md` | Structure complete; Windows result population required for final closure. |

## Status transition policy (embedding models)

- **planned → experimental** requires:
  1. full-model load on CPU with real weights and tokenizer,
  2. passing real-model semantic sanity checks,
  3. documented limitations in `SUPPORTED_MODELS.md`.
- **experimental → shipped** requires:
  1. CPU + DirectML real-model parity at acceptance threshold,
  2. `embed` + `embedBatch` validation coverage,
  3. benchmark evidence and docs updated,
  4. release smoke-run evidence on Windows.

## Explicit non-goals for #39 closure

- Do **not** add decoder/summarizer models to `embed`.
- Do **not** close #39 based only on registry metadata or docs.
- Do **not** treat CPU as debug-only; CPU remains a supported path.
