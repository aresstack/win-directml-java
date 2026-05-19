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
| `intfloat/e5-large-v2`                   | no      | ✅        | ✅             | WordPiece (`vocab.txt`) | mean + L2 + `query: `/`passage: ` prefix                | 🧪 experimental | `directml-encoder` (`E5Encoders.LARGE_V2`)                       |
| `intfloat/multilingual-e5-*`             | no      | ❌        | ❌             | SentencePiece           | mean + L2 + prefix                                      | 🚧 planned      | –                                                                |
| `nomic-ai/nomic-embed-text-v1.5`         | no      | ❌        | ❌             | WordPiece               | mean + L2 + `search_query:` / `search_document:` prefix | 🚧 planned      | –                                                                |

All shipped embedding models go through the same
`com.aresstack.windirectml.encoder.EmbeddingModel` interface and the
batched `embedBatch(...)` path with bucket-padded sequence lengths.
DirectML pooling and L2 normalisation are GPU-resident; only the final
`[N, hidden]` matrix is read back to the host.

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

CPU-only is supported for parity testing and for callers that can
tolerate the ~30× slowdown vs DirectML — every embedding and reranker
ships with a `CpuXxxEncoder` reference implementation that is
byte-identical to the GPU path on the per-element tolerance documented
in the test suite.

## 6. Roadmap

Items here are explicit non-goals of the current release and tracked
for a future minor:

- SentencePiece tokenizer support (blocks multilingual-E5).
- Nomic-text-v1.5 with its custom positional embedding scheme.
- Quantized weights for the BERT encoder family (INT8 GEMM via DML).
- Speculative / batched decoding for Phi-3.

