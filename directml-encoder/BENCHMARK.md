# Embedding benchmark matrix (Issue #39)

This benchmark document tracks **supported embedding models only**.

## Scope

- APIs:
  - `embed` (sequential per-text path)
  - `embedBatch` (batched path)
- Batch sizes:
  - `N=10`
  - `N=50`
  - `N=100`
- Backends:
  - CPU
  - DirectML

Supported embedding models in scope:

- `sentence-transformers/all-MiniLM-L6-v2`
- `danielheinz/e5-base-sts-en-de`
- `intfloat/e5-small-v2`
- `intfloat/e5-base-v2`
- `intfloat/e5-large-v2` (experimental)

Excluded from this benchmark matrix:

- planned/unimplemented embeddings
  (`jinaai/jina-embeddings-v2-base-de`,
  `intfloat/multilingual-e5-large-instruct`,
  `nomic-ai/nomic-embed-text-v1.5`)
- non-embedding models

## Reproducible protocol

### Model prerequisites

```powershell
pwsh scripts/download-minilm.ps1
pwsh scripts/download-e5.ps1 -Variant base-sts-en-de
pwsh scripts/download-e5.ps1 -Variant small-v2
pwsh scripts/download-e5.ps1 -Variant base-v2
pwsh scripts/download-e5.ps1 -Variant large-v2
```

### Benchmark command

`EmbedBatchBenchmark` reports both:

- sequential `embed` aggregate timing (`loopMs`, `loopPerMs`)
- `embedBatch` timing (`batchMs`, `batchPerMs`)

```powershell
./gradlew :directml-encoder:runEmbedBatchBenchmark --args="model both both 1 10,50,100"
```

Variant-specific run examples:

```powershell
./gradlew :directml-encoder:runEmbedBatchBenchmark --args="model minilm both 1 10,50,100"
./gradlew :directml-encoder:runEmbedBatchBenchmark --args="model e5 cpu 1 10,50,100"
./gradlew :directml-encoder:runEmbedBatchBenchmark --args="model e5 dml 1 10,50,100"
```

### Required metadata to log with each run

- date/time
- OS version
- Java version
- GPU + driver
- `DirectML.dll` source/version
- benchmark command line
- selected model variant

## Results template

> Fill this table from Windows benchmark runs. Keep one row per `(model, backend, API, N)`.

| model | status | backend | API | N | total ms | per-text ms | notes |
|---|---|---|---|---:|---:|---:|---|
| all-MiniLM-L6-v2 | shipped | cpu | embed | 10 | TBD | TBD | |
| all-MiniLM-L6-v2 | shipped | cpu | embedBatch | 10 | TBD | TBD | |
| all-MiniLM-L6-v2 | shipped | directml | embed | 10 | TBD | TBD | |
| all-MiniLM-L6-v2 | shipped | directml | embedBatch | 10 | TBD | TBD | |
| all-MiniLM-L6-v2 | shipped | cpu | embed | 50 | TBD | TBD | |
| all-MiniLM-L6-v2 | shipped | cpu | embedBatch | 50 | TBD | TBD | |
| all-MiniLM-L6-v2 | shipped | directml | embed | 50 | TBD | TBD | |
| all-MiniLM-L6-v2 | shipped | directml | embedBatch | 50 | TBD | TBD | |
| all-MiniLM-L6-v2 | shipped | cpu | embed | 100 | TBD | TBD | |
| all-MiniLM-L6-v2 | shipped | cpu | embedBatch | 100 | TBD | TBD | |
| all-MiniLM-L6-v2 | shipped | directml | embed | 100 | TBD | TBD | |
| all-MiniLM-L6-v2 | shipped | directml | embedBatch | 100 | TBD | TBD | |
| e5-base-sts-en-de | shipped | cpu | embed | 10 | TBD | TBD | |
| e5-base-sts-en-de | shipped | cpu | embedBatch | 10 | TBD | TBD | |
| e5-base-sts-en-de | shipped | directml | embed | 10 | TBD | TBD | |
| e5-base-sts-en-de | shipped | directml | embedBatch | 10 | TBD | TBD | |
| e5-base-sts-en-de | shipped | cpu | embed | 50 | TBD | TBD | |
| e5-base-sts-en-de | shipped | cpu | embedBatch | 50 | TBD | TBD | |
| e5-base-sts-en-de | shipped | directml | embed | 50 | TBD | TBD | |
| e5-base-sts-en-de | shipped | directml | embedBatch | 50 | TBD | TBD | |
| e5-base-sts-en-de | shipped | cpu | embed | 100 | TBD | TBD | |
| e5-base-sts-en-de | shipped | cpu | embedBatch | 100 | TBD | TBD | |
| e5-base-sts-en-de | shipped | directml | embed | 100 | TBD | TBD | |
| e5-base-sts-en-de | shipped | directml | embedBatch | 100 | TBD | TBD | |
| e5-small-v2 | shipped | cpu | embed | 10 | TBD | TBD | |
| e5-small-v2 | shipped | cpu | embedBatch | 10 | TBD | TBD | |
| e5-small-v2 | shipped | directml | embed | 10 | TBD | TBD | |
| e5-small-v2 | shipped | directml | embedBatch | 10 | TBD | TBD | |
| e5-small-v2 | shipped | cpu | embed | 50 | TBD | TBD | |
| e5-small-v2 | shipped | cpu | embedBatch | 50 | TBD | TBD | |
| e5-small-v2 | shipped | directml | embed | 50 | TBD | TBD | |
| e5-small-v2 | shipped | directml | embedBatch | 50 | TBD | TBD | |
| e5-small-v2 | shipped | cpu | embed | 100 | TBD | TBD | |
| e5-small-v2 | shipped | cpu | embedBatch | 100 | TBD | TBD | |
| e5-small-v2 | shipped | directml | embed | 100 | TBD | TBD | |
| e5-small-v2 | shipped | directml | embedBatch | 100 | TBD | TBD | |
| e5-base-v2 | shipped | cpu | embed | 10 | TBD | TBD | |
| e5-base-v2 | shipped | cpu | embedBatch | 10 | TBD | TBD | |
| e5-base-v2 | shipped | directml | embed | 10 | TBD | TBD | |
| e5-base-v2 | shipped | directml | embedBatch | 10 | TBD | TBD | |
| e5-base-v2 | shipped | cpu | embed | 50 | TBD | TBD | |
| e5-base-v2 | shipped | cpu | embedBatch | 50 | TBD | TBD | |
| e5-base-v2 | shipped | directml | embed | 50 | TBD | TBD | |
| e5-base-v2 | shipped | directml | embedBatch | 50 | TBD | TBD | |
| e5-base-v2 | shipped | cpu | embed | 100 | TBD | TBD | |
| e5-base-v2 | shipped | cpu | embedBatch | 100 | TBD | TBD | |
| e5-base-v2 | shipped | directml | embed | 100 | TBD | TBD | |
| e5-base-v2 | shipped | directml | embedBatch | 100 | TBD | TBD | |
| e5-large-v2 | experimental | cpu | embed | 10 | TBD | TBD | |
| e5-large-v2 | experimental | cpu | embedBatch | 10 | TBD | TBD | |
| e5-large-v2 | experimental | directml | embed | 10 | TBD | TBD | |
| e5-large-v2 | experimental | directml | embedBatch | 10 | TBD | TBD | |
| e5-large-v2 | experimental | cpu | embed | 50 | TBD | TBD | |
| e5-large-v2 | experimental | cpu | embedBatch | 50 | TBD | TBD | |
| e5-large-v2 | experimental | directml | embed | 50 | TBD | TBD | |
| e5-large-v2 | experimental | directml | embedBatch | 50 | TBD | TBD | |
| e5-large-v2 | experimental | cpu | embed | 100 | TBD | TBD | |
| e5-large-v2 | experimental | cpu | embedBatch | 100 | TBD | TBD | |
| e5-large-v2 | experimental | directml | embed | 100 | TBD | TBD | |
| e5-large-v2 | experimental | directml | embedBatch | 100 | TBD | TBD | |

## Existing measured reference points

From Windows smoke evidence (`WINDOWS-SMOKE-RUN.md`):

- MiniLM DirectML `embedBatch` at `N=32`: `~10.146 ms/text` on RTX 5080

Keep this as a coarse anchor only; use the matrix above for the official
N=10/50/100 comparison set.

## Recommendation section (to be finalized with full matrix data)

- **Small default model**
  - Prefer `sentence-transformers/all-MiniLM-L6-v2` for low-latency default use.
- **Good German model**
  - Prefer `danielheinz/e5-base-sts-en-de` when en/de sentence similarity quality is primary.
- **Multilingual option**
  - Current shipped multilingual-ready path is limited; `multilingual-e5-large-instruct` remains planned.
- **CPU-only suitability**
  - CPU path is supported and valid for local/smaller workloads.
- **When DirectML is recommended**
  - Use DirectML for sustained ingestion/retrieval workloads and larger batches where throughput dominates.
