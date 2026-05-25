# Issue #39 — Acceptance Criteria Matrix

> _Part of [#39](https://github.com/aresstack/win-directml-java/issues/39):
> Firmen-Modellliste für Embeddings sauber unterstützen und klassifizieren._
>
> This file records the final acceptance-criteria check after the #39
> follow-up work landed. It is deliberately an evidence matrix: it points
> to the code, tests and documentation that define the current contract,
> and it separates shipped runtime support from deliberately planned
> future model families.

## Sub-issues and merged evidence

| Sub-issue | Topic | Evidence status |
|-----------|-------|-----------------|
| #39-A | MiniLM / E5 WordPiece runtime validation | done — registry, sidecar gate and shipped WordPiece runtime are in `main` |
| #39-B / #51 | `jinaai/jina-embeddings-v2-base-de` analysis | done — Jina is `planned`, documented in `SUPPORTED_MODELS.md` §1.1.2 and pinned in registry tests |
| #39-C / #52 | `intfloat/multilingual-e5-large-instruct` analysis | done — multilingual-E5 is `planned`, documented in `SUPPORTED_MODELS.md` §1.1.1 and pinned in registry tests |
| #39-D / #53 | Benchmark matrix | done — `BENCHMARK.md` covers shipped embedding models and documents planned-model exclusions |
| #39-E / #54 | Final acceptance-criteria matrix | this file |

Merged PR evidence includes #45, #48, #50, #56, #58, #59 and #60.

## Status legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Criterion is satisfied for the current release contract. |
| 🧪 | Runtime path exists, but some evidence is environment-bound or explicitly partial. |
| 🚧 | Deliberately planned/future work; the release contract is satisfied by clear classification and rejection, not by shipped runtime support. |
| ⛔ | Not satisfied. No criterion in this matrix is in this state. |

## Acceptance criteria

| # | Criterion | Status | Evidence |
|---|-----------|:------:|----------|
| 1 | Central registry contains all company-list models. | ✅ | `EmbeddingModelRegistry` contains all seven model IDs and is shared from `directml-config` so sidecar/workbench/client use one Java-8-compatible source of truth. `EmbeddingModelRegistryTest.allSevenCompanyModelsArePresent` pins this. |
| 2 | Embedding models are correctly classified as `shipped`, `experimental` or `planned`. | ✅ | `SUPPORTED_MODELS.md` §1 and §1.1 match the registry. MiniLM, `intfloat/e5-small-v2` and `intfloat/e5-base-v2` are shipped; `intfloat/e5-large-v2` remains experimental; `danielheinz/e5-base-sts-en-de`, multilingual-E5 and Jina are planned. |
| 3 | Non-embedding models are not offered through the `embed` path. | ✅ | `DirectMlPhi3Sidecar.embedFamily(...)` resolves full model IDs via the registry and rejects decoder/summarizer IDs with use-case-specific errors. `EmbedFamilyParseTest` and registry tests pin the messages. |
| 4 | Workbench model selection is use-case aware. | ✅ | Workbench embedding options are populated from `EmbeddingModelRegistry.entriesByUseCase(EMBEDDING)`, so decoder/summarizer IDs are filtered out before the sidecar gate is reached. |
| 5 | `sentence-transformers/all-MiniLM-L6-v2` works locally with CPU and DirectML. | ✅ | MiniLM has shipped CPU and DirectML implementations, registry `embedFamily=minilm`, shipped status, and real benchmark evidence in `BENCHMARK.md`. |
| 6 | `danielheinz/e5-base-sts-en-de` is either real-validated or clearly not shipped. | ✅ | The upstream checkpoint was validated as XLM-R/SentencePiece rather than the current WordPiece E5 profile. It is now `Status.PLANNED`, `embedFamily=null`, documented in `SUPPORTED_MODELS.md` §1.0.1 and routed to the future SentencePiece/XLM-R track. |
| 7 | `jinaai/jina-embeddings-v2-base-de` is analysed and correctly classified. | ✅ | #60 records JinaBERT v2 as planned: ALiBi + GLU-style MLP mean it is not a drop-in for the current BERT/MiniLM/E5 core. `SUPPORTED_MODELS.md` §1.1.2 and `jinaV2EntryCarriesAnalysisFields` pin this. |
| 8 | `intfloat/multilingual-e5-large-instruct` is analysed and correctly classified. | ✅ | #59 records it as planned: SentencePiece + XLM-RoBERTa-large + instruction-prefix handling are required before runtime support. `SUPPORTED_MODELS.md` §1.1.1 and `multilingualE5InstructAnalysisIsPinned` pin this. |
| 9 | `SUPPORTED_MODELS.md` is current and authoritative. | ✅ | It contains the shipped/experimental/planned model table, sidecar/workbench classification contract, explicit rejection behavior, multilingual-E5 analysis and Jina analysis. |
| 10 | `BENCHMARK.md` contains the benchmark matrix or honestly records missing measurements. | ✅ | #58 adds the top-level benchmark matrix for shipped embedding runtime models. It keeps planned models out of the measured matrix and documents planned-model exclusions instead of inventing numbers. |
| 11 | CPU is documented as a supported local path, not just a debug/reference mode. | ✅ | `SUPPORTED_MODELS.md` §5 states CPU backends are supported as local fallback/smaller-workload paths. Shipped embedding/reranker implementations include CPU encoder classes and matching tests. |
| 12 | Embeddings do not depend on a cloud service at runtime. | ✅ | Runtime uses local model directories and local CPU/DirectML execution. Model downloads are out-of-band setup helpers; the sidecar does not fetch checkpoints from the network at runtime. |

## Documentation coverage

| Document | Coverage |
|----------|----------|
| `SUPPORTED_MODELS.md` | Authoritative support/classification document for the company model list, shipped models, planned models and sidecar rejection semantics. |
| `BENCHMARK.md` | Shipped embedding benchmark matrix and recommendations. Planned models remain excluded until runtime support exists. |
| `docs/concept-sentencepiece-xlmr.md` | Future SentencePiece/XLM-R implementation plan for multilingual-E5 and related XLM-R-family models. |
| `docs/concept-jina.md` | Future JinaBERT/Jina embeddings implementation plan. |
| `WORKBENCH.md` | User-facing workbench behavior and local model selection. |
| `README.md` | Main user path and cross-links into supported models and benchmarks. |

## Test evidence commands

The following commands are the relevant validation set for the #39 surface:

```bash
./gradlew :directml-config:test
./gradlew :directml-sidecar-workbench:test
./gradlew :directml-sidecar:test --tests '*EmbedFamilyParseTest'
./gradlew publishToMavenLocal -Pversion=0.1.0-beta.1
```

Additional real-model and benchmark runs are hardware/model-directory bound:

```powershell
./gradlew.bat :directml-encoder:test --tests '*MiniLm*RealModel*'
./gradlew.bat :directml-encoder:test --tests '*E5RealModelReferenceTest'
./gradlew.bat :directml-encoder:test --tests '*RerankerRealModelReferenceTest'
./gradlew.bat :directml-encoder:runEmbedBatchBenchmark --args="model minilm both 1 10,50,100"
./gradlew.bat :directml-encoder:runRerankerBenchmark
```

The current release contract does **not** require Jina or multilingual-E5
real-model runs because both are explicitly classified as planned and rejected
by the current sidecar runtime.

## Final status for #39

All twelve acceptance criteria are satisfied for the current release contract.
The final state is intentionally conservative:

- shipped runtime support is claimed only for models with implemented and tested runtime paths;
- planned models are visible in the registry/workbench classification but rejected at runtime with clear messages;
- benchmark documentation does not invent measurements for planned models;
- future implementation work is split into SentencePiece/XLM-R and Jina concept tracks.

This PR is the final #39-E evidence matrix. It should be merged with a normal
`Part of #39` reference; #39 can be closed separately after confirming that no
open follow-up acceptance item remains.
