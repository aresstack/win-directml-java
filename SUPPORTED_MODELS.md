# Supported models

This file is the authoritative list of model checkpoints that
win-directml-java ships **runtime support for** in the published
artifacts. The model **weights** themselves are not redistributed
(see [`MODEL_LICENSES.md`](MODEL_LICENSES.md)); use the
`scripts/download-*.ps1` helpers to fetch them.

Status legend:

- ✅ **shipped** — production-ready, has a CPU reference + DirectML
  parity test in the test suite and is exercised by the per-release
  smoke run.
- 🧪 **experimental** — works end-to-end, but feature coverage or
  numerical tolerance is not yet release-grade.
- 🚧 **planned** — runtime support is in progress / partly implemented.

## 1. Embedding models

| Variant                                  | Default | DirectML | CPU reference | Tokenizer               | Pooling                                                 | Status          | Module                                                           |
|------------------------------------------|---------|----------|---------------|-------------------------|---------------------------------------------------------|-----------------|------------------------------------------------------------------|
| `sentence-transformers/all-MiniLM-L6-v2` | yes     | ✅        | ✅             | WordPiece (`vocab.txt`) | mean + optional L2                                      | ✅ shipped       | `directml-encoder` (`DirectMlMiniLmEncoder`, `CpuMiniLmEncoder`) |
| `intfloat/e5-small-v2`                   | no      | ✅        | ✅             | WordPiece (`vocab.txt`) | mean + L2 + `query: `/`passage: ` prefix                | ✅ shipped       | `directml-encoder` (`E5Encoders.SMALL_V2`)                       |
| `intfloat/e5-base-v2`                    | no      | ✅        | ✅             | WordPiece (`vocab.txt`) | mean + L2 + `query: `/`passage: ` prefix                | ✅ shipped       | `directml-encoder` (`E5Encoders.BASE_V2`)                        |
| `danielheinz/e5-base-sts-en-de`          | no      | ✅        | ✅             | WordPiece (`vocab.txt`) | mean + L2 + `query: `/`passage: ` prefix                | ✅ shipped       | `directml-encoder` (`E5Encoders.BASE_STS_EN_DE`)                 |
| `intfloat/e5-large-v2`                   | no      | ✅        | ✅             | WordPiece (`vocab.txt`) | mean + L2 + `query: `/`passage: ` prefix                | 🧪 experimental | `directml-encoder` (`E5Encoders.LARGE_V2`)                       |
| `intfloat/multilingual-e5-large-instruct`| no      | ❌        | ❌             | SentencePiece (XLM-R)   | mean + L2 + `Instruct: …\nQuery: …` prefix              | 🚧 planned      | – (needs SentencePiece + XLM-RoBERTa core)                       |
| `jinaai/jina-embeddings-v2-base-de`      | no      | ❌        | ❌             | WordPiece (Jina-custom) | mean + L2; ALiBi positional bias                        | 🚧 planned      | – (needs custom Jina v2 attention path)                          |
| `nomic-ai/nomic-embed-text-v1.5`         | no      | ❌        | ❌             | WordPiece               | mean + L2 + `search_query:` / `search_document:` prefix | 🚧 planned      | –                                                                |

All shipped embedding models go through the same
`com.aresstack.windirectml.encoder.EmbeddingModel` interface and the
batched `embedBatch(...)` path with bucket-padded sequence lengths.
DirectML pooling and L2 normalisation are GPU-resident; only the final
`[N, hidden]` matrix is read back to the host.

### 1.1 In-house model list classification

The embedding-pipeline classification of the in-house model catalogue is
mirrored in `EmbeddingModelRegistry` (module `directml-sidecar`). The
registry is the single source of truth for the `embed` gate: passing any
of the model IDs below to `-Dembed.model` resolves through the registry,
so decoder / summarizer IDs are rejected with an explicit
*"… is not an embedding model …"* error rather than being silently
treated as an unknown family.

| `modelId`                                   | `useCase`  | `status`        | Backend support   | Notes                                                                                                                |
|---------------------------------------------|------------|-----------------|-------------------|----------------------------------------------------------------------------------------------------------------------|
| `sentence-transformers/all-MiniLM-L6-v2`    | embedding  | ✅ shipped       | CPU + DirectML    | Default fast embedding model (WordPiece, BERT-style).                                                                |
| `danielheinz/e5-base-sts-en-de`             | embedding  | ✅ shipped       | CPU + DirectML    | German/English STS fine-tune; uses `"query: "` / `"passage: "` E5 prefixes.                                          |
| `intfloat/multilingual-e5-large-instruct`   | embedding  | 🚧 planned      | – (planned)       | NOT compatible with the current WordPiece-only E5 path. Requires SentencePiece + XLM-RoBERTa core.                   |
| `jinaai/jina-embeddings-v2-base-de`         | embedding  | 🚧 planned      | – (planned)       | Jina BERT v2 uses ALiBi positional bias; not a drop-in for the standard BERT core. Requires analysis before shipping.|
| `openai/gpt-oss-120b`                       | decoder    | ⛔ unsupported  | – (not for embed) | Decoder-only LLM. Rejected by the `embed` endpoint.                                                                  |
| `casperhansen/llama-3.3-70b-instruct-awq`   | decoder    | ⛔ unsupported  | – (not for embed) | Llama 3.3 70B AWQ-quantised decoder-only LLM. Rejected by the `embed` endpoint.                                      |
| `ellamind/summarizer-v6-llama-v2`           | summarizer | ⛔ unsupported  | – (not for embed) | Llama-v2 summarizer fine-tune. Belongs to a future text-generation/summarize ticket, not the embed endpoint.         |

Passing a decoder / summarizer ID to `-Dembed.model` fails with the
following exact message (matched by both the registry test suite and any
downstream tooling that parses sidecar errors):

```
Model openai/gpt-oss-120b is not an embedding model. Decoder models are not supported by the embed endpoint.
```

Passing an embedding ID that is currently `planned` (Jina v2,
multilingual-E5-instruct) fails with a status-aware message that points
at this file:

```
Model jinaai/jina-embeddings-v2-base-de is classified as an embedding model but has no runtime support in this build (status=planned). See SUPPORTED_MODELS.md for the current classification.
```


## 2. Reranker models

| Variant                                | Default | DirectML | CPU reference | Tokenizer | Architecture                                | Status    | Module                                                 |
|----------------------------------------|---------|----------|---------------|-----------|---------------------------------------------|-----------|--------------------------------------------------------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | yes     | ✅        | ✅             | WordPiece | BERT cross-encoder + linear regression head | ✅ shipped | `directml-encoder` (`DirectMlReranker`, `CpuReranker`) |

Rerankers share the BERT encoder stack with the embedding pipeline,
including the per-layer command-list coalescing (see
`directml-encoder/BENCHMARK.md`).

## 3. LLM / decoder models

| Variant                                                       | Quantization    | DirectML | Greedy decode | KV cache | Status          | Module               |
|---------------------------------------------------------------|-----------------|----------|---------------|----------|-----------------|----------------------|
| `microsoft/Phi-3-mini-4k-instruct-onnx` (DirectML INT4 build) | INT4 GroupQuant | ✅        | ✅             | paged    | 🧪 experimental | `directml-inference` |

The Phi-3 pipeline runs prefill and decode in a single DirectML graph
per layer; speculative decoding, batched generation and beam search are
out of scope for the current release. The Phi-3 model weights are not
permissively licensed — read
https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx
before redistributing anything you build on top of them.

> **Summarization is experimental.** The `summarize` JSON-RPC method is
> backed exclusively by the Phi-3 runtime above and is **not part of the
> first Maven Central core release**. `directml-inference` is sidecar-only
> for `0.1.0-beta.1` and is *not* published to Central. Embeddings and
> reranking are the primary supported release use cases. Decoder models
> are not part of the published core artifacts. When no Phi-3 model
> directory is present the sidecar starts cleanly and `summarize` replies
> with `-32005 Not implemented`; when files are missing the
> `sidecar.modelLoadFailed` notification names the specific missing file
> (e.g. `Phi-3 model directory is missing tokenizer.json`).
> CPU-only summarization is supported by the Phi-3 runtime, but the
> intended acceleration path is DirectML – pass `-Dphi3.backend=auto`
> (default) or `-Dphi3.backend=directml`; `-Dphi3.backend=cpu` works as a
> local fallback.

## 4. Sidecar / JSON-RPC

The `directml-sidecar` module exposes the embedding, reranking and
text-generation endpoints over a local JSON-RPC port (no network bind
by default). The pure-Java-8 client (`directml-sidecar-client-java8`,
also on Maven Central) is the supported way for Java-8 host
applications to call the sidecar. See
[`directml-sidecar/PROTOCOL.md`](directml-sidecar/PROTOCOL.md) for the
wire format.

## 5. Hardware requirements

| Requirement                        | Why                                                                                                       |
|------------------------------------|-----------------------------------------------------------------------------------------------------------|
| Windows 10 21H1+ / Windows 11      | DirectML in-box (1.8+); side-by-side `DirectML.dll` is also supported via `-Dwindirectml.directml.dll=…`. Some optional fast paths use newer DirectML feature levels when available. |
| A DirectX-12-capable GPU           | every shipped backend dispatches via D3D12 + DirectML                                                     |
| Feature level ≥ DirectML 1.0 / 2.0 | all encoder kernels stay on the FL 1.0 / 2.0 baseline (composite GELU fallback for FL < 5.1)              |
| ≥ 2 GB free GPU memory             | comfortable headroom for E5-base + bucketed batch padding                                                 |
| Java 21 (host)                     | code uses the FFM API                                                                                     |
| Java 8 (sidecar client only)       | the sidecar client artifact is compiled with `-release 8`                                                 |

CPU backends are supported as a local fallback and for smaller local
workloads — every embedding and reranker ships with a `CpuXxxEncoder`
implementation that is byte-identical to the GPU path on the per-element
tolerance documented in the test suite, making it suitable for use
without a DirectML-capable GPU.

## 6. Roadmap

Items here are explicit non-goals of the current release and tracked
for a future minor:

- SentencePiece tokenizer support (blocks multilingual-E5).
- Nomic-text-v1.5 with its custom positional embedding scheme.
- Quantized weights for the BERT encoder family (INT8 GEMM via DML).
- Speculative / batched decoding for Phi-3.

