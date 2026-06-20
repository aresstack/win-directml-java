# Embedding Benchmark Matrix

This document is the reproducible, top-level benchmark reference for the
**shipped embedding models** in `win-directml-java`. It complements the
shipping list in [`SUPPORTED_MODELS.md`](SUPPORTED_MODELS.md) and the
deeper kernel-level write-up in
[`directml-encoder/BENCHMARK.md`](directml-encoder/BENCHMARK.md).

Scope follows the model registry in `directml-config`
(`EmbeddingModelRegistry`, `useCase=EMBEDDING`): only models that have
**status = shipped** and a real CPU + DirectML runtime path are
benchmarked. Models classified as 🚧 *planned* or 🧪 *experimental*
are **not** benchmarked here.

> **Excluded from benchmark scope:**
>
> - `danielheinz/e5-base-sts-en-de` — classified as 🚧 *planned*
    > since PR #56: the upstream checkpoint is an XLM-R / SentencePiece
    > multilingual-E5 derivative and is incompatible with the current
    > WordPiece-only E5 runtime. No runtime path, no benchmark.
> - `intfloat/e5-large-v2` — 🧪 *experimental*; not yet release-grade.
> - `jinaai/jina-embeddings-v2-base-de`, `intfloat/multilingual-e5-large-instruct`,
    > `nomic-ai/nomic-embed-text-v1.5` — 🚧 *planned*; no runtime path.

> **Measurements vs. recommendations.** Section [§4](#4-results) contains
> measured numbers. The derived guidance lives in a clearly separated
> [§5](#5-recommendations) block and is never mixed with the raw tables.

---

## 1. Benchmark scope

| Model                                    | Embed family | Backends benchmarked | Status    | Source / download             |
|------------------------------------------|--------------|----------------------|-----------|-------------------------------|
| `sentence-transformers/all-MiniLM-L6-v2` | `minilm`     | CPU + DirectML       | ✅ shipped | Workbench Download tab |
| `intfloat/e5-small-v2`                   | `e5`         | CPU + DirectML       | ✅ shipped | Workbench Download tab |
| `intfloat/e5-base-v2`                    | `e5`         | CPU + DirectML       | ✅ shipped | Workbench Download tab |

For each `(backend, model)` pair the benchmark sweeps:

- methods: `embed` (sequential per-text loop) and `embedBatch` (one
  bucket-padded batch call);
- batch sizes: `N ∈ {10, 50, 100}`;
- backends: `cpu` (Java + DirectML disabled), `dml` (DirectML on a
  DX12 GPU).

The harness measures both calls in the **same** run, so the
per-text-`embed` cost (`loopPerMs`) and the per-text-`embedBatch` cost
(`batchPerMs`) are directly comparable on the same host.

## 2. Benchmark protocol

Each table below was produced under the following protocol. **Do not
fold these fields into the recommendations**; they are the data-
provenance contract for every measured row.

| Field             | Value                                                                                                                                                                                          |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Hardware          | See per-result-table "Host" line below. CPU column = same host, DirectML disabled.                                                                                                             |
| OS                | Windows 11, Build 22H2+ (DirectML in-box ≥ 1.8).                                                                                                                                               |
| Java (sidecar)    | Java 21 with `--enable-preview` (FFM API). Java 8 modules build with `-release 8`.                                                                                                             |
| DirectML          | Windows-in-box `DirectML.dll` (FL 2.0 / 5.0 fast path where available). Side-by-side `DirectML.dll` is also supported via `-Dwindirectml.directml.dll=…`.                                      |
| Model directory   | `model/all-MiniLM-L6-v2/` (MiniLM), `model/e5-small-v2/` / `model/e5-base-v2/` (E5).                                                                                                           |
| Download path     | Workbench Download tab.                                                                                                                                                                      |
| Warmup            | 1 × `embedBatch(maxN)` so every pad-bucket entry is hot before the first measured `N`.                                                                                                         |
| Repetitions       | 1 timed run per `N` per `(method × backend × model)`. The harness is intentionally simple — no JMH, no statistical smoothing — it has to make order-of-magnitude differences visible, not 5 %. |
| Measurement       | `loopMs` = wall time of `N` × `embed(r)`; `batchMs` = wall time of one `embedBatch(reqs)`; per-text cost in `ms/text` is `loopMs/N` and `batchMs/N`.                                           |
| Corpus            | 4 round-robined templates (2 short / 2 long), spans pad buckets `S ∈ {64, 128}`.                                                                                                               |
| Known variance    | Single-shot timings include JIT/IO noise; expect ±5–10 % on cold runs. DirectML cold-start dominates the first measured `N` if warmup = 0.                                                     |
| Known limitations | Pad-bucket cache is keyed on `S ∈ {64, 128, 256, 512}` — corpora outside those buckets recompile a stack on first hit and inflate that single row.                                             |

## 3. Reproducible commands

The model weights are **not** redistributed (see
[`MODEL_LICENSES.md`](MODEL_LICENSES.md)); fetch them once with the
scripts under `scripts/`. All commands are run from the repository
root.

```powershell
# 1. Download supported embedding model weights into ./model/...
Use the Workbench Download tab to fetch MiniLM and E5 variants before running the benchmark.
```

```bash
# 2. Build + run the embedding-only test suites that the issue lists as
#    the minimum proof of work for the embedding pipeline.
./gradlew :directml-config:test
./gradlew :directml-sidecar-workbench:test
./gradlew publishToMavenLocal -Pversion=0.1.0-beta.1

# 3. Run the embed / embedBatch benchmark. Args are positional:
#       <modelRoot> <which> <backend> <warmup> <sizes-csv>
#    where `which ∈ {minilm,e5,both}`, `backend ∈ {cpu,dml,both}`.

# MiniLM, CPU + DirectML, N = 10, 50, 100, 1 warmup batch:
./gradlew :directml-encoder:runEmbedBatchBenchmark \
    --args="model minilm both 1 10,50,100"

# E5 small-v2, CPU + DirectML, N = 10, 50, 100:
./gradlew :directml-encoder:runEmbedBatchBenchmark \
    --args="model e5 both 1 10,50,100" -De5.model=small-v2

# E5 base-v2, CPU + DirectML, N = 10, 50, 100:
./gradlew :directml-encoder:runEmbedBatchBenchmark \
    --args="model e5 both 1 10,50,100" -De5.model=base-v2
```

The harness skips any backend or model directory that is not present on
the host (it prints `[skip] dml/e5: ...` instead of failing) so the same
command line works on a CPU-only Linux dev box and on a Windows
DirectML host.

## 4. Results — `sentence-transformers/all-MiniLM-L6-v2`

Host: Windows 11, DirectML in-box `DirectML.dll` 1.8.0, MiniLM weights
under `model/all-MiniLM-L6-v2/`, warmup = 1 × N=100 batch, repetitions
= 1, corpus = 4-template round-robin spanning pad buckets `{64, 128}`.
These are the same numbers as the "V2.1 — per-layer-coalesced" run in
[`directml-encoder/BENCHMARK.md`](directml-encoder/BENCHMARK.md) and
were produced by:

```
gradle :directml-encoder:runEmbedBatchBenchmark --args="model minilm both 1 10,50,100"
```

### 4.1 CPU backend

| Method       |   N | wall (ms) | per-text (ms/text) | throughput (texts/s) |
|--------------|----:|----------:|-------------------:|---------------------:|
| `embed` ×N   |  10 |  2 505.76 |             250.58 |                 3.99 |
| `embedBatch` |  10 |  2 522.01 |             252.20 |                 3.97 |
| `embed` ×N   |  50 | 14 016.34 |             280.33 |                 3.57 |
| `embedBatch` |  50 | 13 800.76 |             276.02 |                 3.62 |
| `embed` ×N   | 100 | 28 288.59 |             282.89 |                 3.54 |
| `embedBatch` | 100 | 28 275.32 |             282.75 |                 3.54 |

### 4.2 DirectML backend

| Method       |   N | wall (ms) | per-text (ms/text) | throughput (texts/s) |
|--------------|----:|----------:|-------------------:|---------------------:|
| `embed` ×N   |  10 |     95.51 |               9.55 |               104.70 |
| `embedBatch` |  10 |     93.69 |               9.37 |               106.74 |
| `embed` ×N   |  50 |    446.99 |               8.94 |               111.86 |
| `embedBatch` |  50 |    435.78 |               8.72 |               114.74 |
| `embed` ×N   | 100 |    881.60 |               8.82 |               113.43 |
| `embedBatch` | 100 |    967.09 |               9.67 |               103.40 |

### 4.3 CPU ↔ DirectML speedup (MiniLM)

| Method       |   N | CPU ms/text | DML ms/text | Speedup |
|--------------|----:|------------:|------------:|--------:|
| `embed` ×N   |  10 |      250.58 |        9.55 |   26.2× |
| `embedBatch` |  10 |      252.20 |        9.37 |   26.9× |
| `embed` ×N   |  50 |      280.33 |        8.94 |   31.4× |
| `embedBatch` |  50 |      276.02 |        8.72 |   31.7× |
| `embed` ×N   | 100 |      282.89 |        8.82 |   32.1× |
| `embedBatch` | 100 |      282.75 |        9.67 |   29.2× |

Observation (not yet a recommendation): on this MiniLM host, the
`embedBatch` and `embed`-loop per-text cost converge once the
pad-bucket cache is hot, because the warmup batch primes both paths.
The DirectML per-text cost is dominated by the
per-encoder-layer command-list submission — see
[`directml-encoder/BENCHMARK.md`](directml-encoder/BENCHMARK.md) for
the kernel-level breakdown.

### 4.4 Results — E5 WordPiece variants (`e5-small-v2`, `e5-base-v2`)

> **Not yet measured on the reference host.** The shipped WordPiece E5
> variants (`intfloat/e5-small-v2`, `intfloat/e5-base-v2`) have a
> working CPU + DirectML runtime path but have not yet been benchmarked
> on the reference Windows host. The commands in §3 are directly
> executable once the model weights are downloaded. Tables will be
> added here when measurements are available.
>
> We are deliberately leaving the result tables empty rather than
> copying values from MiniLM, per the issue's *"Keine Benchmark-Werte
> ohne echten Lauf"* rule.

---

## 5. Recommendations

> The numbers in §4 are the only inputs that back the guidance below;
> everything in this section is an interpretation and is **not** a
> measurement. CPU is a fully supported local backend, not a debug
> fallback.

| Use case                              | Recommended model                        | Recommended backend                  | Rationale (from §4)                                                                                                                                                                |
|---------------------------------------|------------------------------------------|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Small / fast default                  | `sentence-transformers/all-MiniLM-L6-v2` | DirectML if available, CPU otherwise | 6-layer encoder, ~9 ms/text on DirectML and ~280 ms/text on CPU (§4); ships as the workbench default.                                                                              |
| German-language / German+English text | _no shipped model_                       | _n/a_                                | `danielheinz/e5-base-sts-en-de` is 🚧 planned (XLM-R/SentencePiece mismatch, see §5.1). No German-optimized model ships today.                                                     |
| Multilingual coverage beyond English  | _none shipped_                           | _n/a_                                | `intfloat/multilingual-e5-large-instruct` is 🚧 *planned* (needs SentencePiece + XLM-R core) and intentionally **not** benchmarked.                                                |
| CPU-only host (no DirectML GPU)       | `sentence-transformers/all-MiniLM-L6-v2` | CPU                                  | ~3.5 texts/s on the reference CPU host with `embedBatch`, byte-identical (within documented tolerance) to the DirectML output — suitable for local pipelines and offline indexing. |
| DirectML host (Windows 11 + DX12 GPU) | `sentence-transformers/all-MiniLM-L6-v2` | DirectML                             | DirectML is ~26–32× faster than CPU on MiniLM (§4.3). E5 WordPiece variants share the same kernel path so a similar speedup is expected once measured.                             |

Notes:

- All "DirectML recommended" rows are conditioned on
  *"if available"*; the project ships a CPU fallback for every
  embedding model and does not require a DirectML-capable GPU.
- These recommendations cover **embedding** only. Reranker
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`) benchmark numbers live in
  [`directml-encoder/BENCHMARK.md`](directml-encoder/BENCHMARK.md);
  the Phi-3 decoder is tracked separately in
  [`directml-inference/BENCHMARK.md`](directml-inference/BENCHMARK.md).
- *Planned* embedding models (Jina v2, multilingual-E5-instruct,
  Nomic v1.5, `danielheinz/e5-base-sts-en-de`) are explicitly out of
  scope here. They have no runtime path and are gated out by
  `EmbeddingModelRegistry`; the sidecar rejects them with a
  status-aware error pointing at
  [`SUPPORTED_MODELS.md`](SUPPORTED_MODELS.md).

### 5.1 Note on `danielheinz/e5-base-sts-en-de`

This model was previously listed as a benchmark target. As of PR #56,
it is classified as 🚧 *planned*:

- The upstream checkpoint at
  `https://huggingface.co/danielheinz/e5-base-sts-en-de` is an
  `XLMRobertaModel` with a SentencePiece tokenizer (`vocab_size=250002`,
  `type_vocab_size=1`, ~1.06 GB `model.safetensors`).
- The current E5 runtime path in `directml-encoder` supports only
  WordPiece / BERT-base models.
- The model is rejected at runtime by the embed gate with:
  *"… classified as an embedding model but has no runtime support in
  this build (status=planned)."*

Until SentencePiece + XLM-R encoder support lands, this model cannot
be benchmarked and is not a valid follow-up task for this document.
