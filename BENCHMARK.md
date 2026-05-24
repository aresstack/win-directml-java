# Embedding Benchmark Matrix

This document is the reproducible, top-level benchmark reference for the
**supported embedding models** in `win-directml-java`. It complements the
shipping list in [`SUPPORTED_MODELS.md`](SUPPORTED_MODELS.md) and the
deeper kernel-level write-up in
[`directml-encoder/BENCHMARK.md`](directml-encoder/BENCHMARK.md).

Scope follows the model registry in `directml-config`
(`EmbeddingModelRegistry`, `useCase=EMBEDDING`): only models that have a
real CPU + DirectML runtime path are benchmarked. Models classified as
🚧 *planned* (`jinaai/jina-embeddings-v2-base-de`,
`intfloat/multilingual-e5-large-instruct`, `nomic-ai/nomic-embed-text-v1.5`)
are **not** benchmarked — they have no runtime path on disk yet and the
project policy is *"Keine Benchmark-Werte ohne echten Lauf"*.

> **Measurements vs. recommendations.** Sections [§4](#4-results) and
> [§5](#5-results-for-danielheinze5-base-sts-en-de) contain measured
> numbers (or an explicit "not yet measured on this host" note). The
> derived guidance lives in a clearly separated [§6](#6-recommendations)
> block and is never mixed with the raw tables.

---

## 1. Benchmark scope

| Model                                    | Embed family | Backends benchmarked | Source / download                                |
|------------------------------------------|--------------|----------------------|--------------------------------------------------|
| `sentence-transformers/all-MiniLM-L6-v2` | `minilm`     | CPU + DirectML       | `scripts/download-minilm.ps1`                    |
| `danielheinz/e5-base-sts-en-de`          | `e5`         | CPU + DirectML       | `scripts/download-e5.ps1` (`base-sts-en-de`)     |

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

| Field                | Value                                                                                       |
|----------------------|---------------------------------------------------------------------------------------------|
| Hardware             | See per-result-table "Host" line below. CPU column = same host, DirectML disabled.          |
| OS                   | Windows 11, Build 22H2+ (DirectML in-box ≥ 1.8).                                            |
| Java (sidecar)       | Java 21 with `--enable-preview` (FFM API). Java 8 modules build with `-release 8`.          |
| DirectML             | Windows-in-box `DirectML.dll` (FL 2.0 / 5.0 fast path where available). Side-by-side `DirectML.dll` is also supported via `-Dwindirectml.directml.dll=…`. |
| Model directory      | `model/all-MiniLM-L6-v2/` (MiniLM), `model/e5-base-sts-en-de/` (E5 STS de/en).              |
| Download script      | `scripts/download-minilm.ps1`, `scripts/download-e5.ps1`.                                   |
| Warmup               | 1 × `embedBatch(maxN)` so every pad-bucket entry is hot before the first measured `N`.      |
| Repetitions          | 1 timed run per `N` per `(method × backend × model)`. The harness is intentionally simple — no JMH, no statistical smoothing — it has to make order-of-magnitude differences visible, not 5 %. |
| Measurement          | `loopMs` = wall time of `N` × `embed(r)`; `batchMs` = wall time of one `embedBatch(reqs)`; per-text cost in `ms/text` is `loopMs/N` and `batchMs/N`. |
| Corpus               | 4 round-robined templates (2 short / 2 long), spans pad buckets `S ∈ {64, 128}`.            |
| Known variance       | Single-shot timings include JIT/IO noise; expect ±5–10 % on cold runs. DirectML cold-start dominates the first measured `N` if warmup = 0. |
| Known limitations    | Pad-bucket cache is keyed on `S ∈ {64, 128, 256, 512}` — corpora outside those buckets recompile a stack on first hit and inflate that single row. |

## 3. Reproducible commands

The model weights are **not** redistributed (see
[`MODEL_LICENSES.md`](MODEL_LICENSES.md)); fetch them once with the
scripts under `scripts/`. All commands are run from the repository
root.

```powershell
# 1. Download supported embedding model weights into ./model/...
powershell -ExecutionPolicy Bypass -File scripts/download-minilm.ps1
powershell -ExecutionPolicy Bypass -File scripts/download-e5.ps1 -Variant base-sts-en-de
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

# E5 base-sts-en-de, CPU + DirectML, N = 10, 50, 100:
./gradlew :directml-encoder:runEmbedBatchBenchmark \
    --args="model e5 both 1 10,50,100"
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
| `embed` ×N   |  10 |   2 505.76 |              250.58 |                 3.99 |
| `embedBatch` |  10 |   2 522.01 |              252.20 |                 3.97 |
| `embed` ×N   |  50 |  14 016.34 |              280.33 |                 3.57 |
| `embedBatch` |  50 |  13 800.76 |              276.02 |                 3.62 |
| `embed` ×N   | 100 |  28 288.59 |              282.89 |                 3.54 |
| `embedBatch` | 100 |  28 275.32 |              282.75 |                 3.54 |

### 4.2 DirectML backend

| Method       |   N | wall (ms) | per-text (ms/text) | throughput (texts/s) |
|--------------|----:|----------:|-------------------:|---------------------:|
| `embed` ×N   |  10 |      95.51 |                9.55 |               104.70 |
| `embedBatch` |  10 |      93.69 |                9.37 |               106.74 |
| `embed` ×N   |  50 |     446.99 |                8.94 |               111.86 |
| `embedBatch` |  50 |     435.78 |                8.72 |               114.74 |
| `embed` ×N   | 100 |     881.60 |                8.82 |               113.43 |
| `embedBatch` | 100 |     967.09 |                9.67 |               103.40 |

### 4.3 CPU ↔ DirectML speedup (MiniLM)

| Method       |   N | CPU ms/text | DML ms/text | Speedup |
|--------------|----:|------------:|------------:|--------:|
| `embed` ×N   |  10 |       250.58 |         9.55 |   26.2× |
| `embedBatch` |  10 |       252.20 |         9.37 |   26.9× |
| `embed` ×N   |  50 |       280.33 |         8.94 |   31.4× |
| `embedBatch` |  50 |       276.02 |         8.72 |   31.7× |
| `embed` ×N   | 100 |       282.89 |         8.82 |   32.1× |
| `embedBatch` | 100 |       282.75 |         9.67 |   29.2× |

Observation (not yet a recommendation): on this MiniLM host, the
`embedBatch` and `embed`-loop per-text cost converge once the
pad-bucket cache is hot, because the warmup batch primes both paths.
The DirectML per-text cost is dominated by the
per-encoder-layer command-list submission — see
[`directml-encoder/BENCHMARK.md`](directml-encoder/BENCHMARK.md) for
the kernel-level breakdown.

## 5. Results — `danielheinz/e5-base-sts-en-de`

> **Not yet measured on the reference host.** The E5 STS de/en weights
> were not present in `model/e5-base-sts-en-de/` on the host that
> produced §4. The runtime path is shipped (`E5Encoders.BASE_STS_EN_DE`,
> CPU + DirectML, parity-tested in `:directml-encoder:test`), so the
> benchmark is **reproducible** with the command below — we are
> deliberately leaving the result table empty rather than copying
> values from a different model, per the issue's *"Keine Benchmark-
> Werte ohne echten Lauf"* rule.

Command:

```
gradle :directml-encoder:runEmbedBatchBenchmark --args="model e5 both 1 10,50,100"
```

### 5.1 CPU backend

| Method       |   N | wall (ms) | per-text (ms/text) | throughput (texts/s) |
|--------------|----:|----------:|-------------------:|---------------------:|
| `embed` ×N   |  10 | _pending_ |          _pending_ |            _pending_ |
| `embedBatch` |  10 | _pending_ |          _pending_ |            _pending_ |
| `embed` ×N   |  50 | _pending_ |          _pending_ |            _pending_ |
| `embedBatch` |  50 | _pending_ |          _pending_ |            _pending_ |
| `embed` ×N   | 100 | _pending_ |          _pending_ |            _pending_ |
| `embedBatch` | 100 | _pending_ |          _pending_ |            _pending_ |

### 5.2 DirectML backend

| Method       |   N | wall (ms) | per-text (ms/text) | throughput (texts/s) |
|--------------|----:|----------:|-------------------:|---------------------:|
| `embed` ×N   |  10 | _pending_ |          _pending_ |            _pending_ |
| `embedBatch` |  10 | _pending_ |          _pending_ |            _pending_ |
| `embed` ×N   |  50 | _pending_ |          _pending_ |            _pending_ |
| `embedBatch` |  50 | _pending_ |          _pending_ |            _pending_ |
| `embed` ×N   | 100 | _pending_ |          _pending_ |            _pending_ |
| `embedBatch` | 100 | _pending_ |          _pending_ |            _pending_ |

When updating this section on a new host, also fill in `Host:` (CPU
model, GPU model, DirectML version) at the top of §4 / §5 so the
provenance contract from §2 is satisfied.

---

## 6. Recommendations

> The numbers in §4–§5 are the only inputs that back the guidance
> below; everything in this section is an interpretation and is **not**
> a measurement. CPU is a fully supported local backend, not a debug
> fallback.

| Use case                                       | Recommended model                        | Recommended backend    | Rationale (from §4–§5)                                                                                       |
|------------------------------------------------|------------------------------------------|------------------------|--------------------------------------------------------------------------------------------------------------|
| Small / fast default                           | `sentence-transformers/all-MiniLM-L6-v2` | DirectML if available, CPU otherwise | 6-layer encoder, ~9 ms/text on DirectML and ~280 ms/text on CPU (§4); ships as the workbench default. |
| German-language / German+English text          | `danielheinz/e5-base-sts-en-de`          | DirectML if available, CPU otherwise | Shipped E5 path with `"query: "` / `"passage: "` prefix policy; STS-fine-tuned for German + English. Throughput is host-specific; capture §5 locally before committing to a backend. |
| Multilingual coverage beyond German + English  | _none shipped_                           | _n/a_                  | `intfloat/multilingual-e5-large-instruct` is 🚧 *planned* (needs SentencePiece + XLM-R core) and intentionally **not** benchmarked. |
| CPU-only host (no DirectML GPU)                | `sentence-transformers/all-MiniLM-L6-v2` | CPU                    | ~3.5 texts/s on the reference CPU host with `embedBatch`, byte-identical (within documented tolerance) to the DirectML output — suitable for local pipelines and offline indexing. |
| DirectML host (Windows 11 + DX12 GPU)          | `sentence-transformers/all-MiniLM-L6-v2` (fast) or `danielheinz/e5-base-sts-en-de` (quality) | DirectML | DirectML is ~26–32× faster than CPU on MiniLM (§4.3). E5 numbers are pending §5; the kernel path is identical to MiniLM so a similar relative speedup is expected — but treat that as expectation, not as a measurement. |

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
  Nomic v1.5) are explicitly out of scope here. They have no runtime
  path and are gated out by `EmbeddingModelRegistry`; the sidecar
  rejects them with a status-aware error pointing at
  [`SUPPORTED_MODELS.md`](SUPPORTED_MODELS.md).
