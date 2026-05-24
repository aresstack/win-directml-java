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

### 1.0.1 MiniLM / E5 runtime selection and validation

- **MiniLM (`sentence-transformers/all-MiniLM-L6-v2`)**
  - Registry status: `shipped` (`embedFamily=minilm`, CPU + DirectML).
  - `-Dembed.model` accepts both alias `minilm` and full model ID
    `sentence-transformers/all-MiniLM-L6-v2`.
  - Works through both `embed` and `embedBatch`.
- **E5 (`danielheinz/e5-base-sts-en-de`)**
  - Registry status: `shipped` (`embedFamily=e5`, CPU + DirectML).
  - `-Dembed.model` accepts both alias `e5` and full model ID
    `danielheinz/e5-base-sts-en-de`.
  - `-De5.model=base-sts-en-de` selects the expected variant
    (default for the E5 path).
  - Query/document input should use E5 prefixes: `"query: "` /
    `"passage: "` (for both `embed` and `embedBatch`).
  - Real-model CPU-vs-DirectML parity target: cosine `> 0.999`
    when real-model tests are runnable locally.
- **Workbench**
  - The `embed.model` dropdown always includes aliases `minilm`, `e5`
    first and then all registry entries with `useCase=embedding`.

### 1.1 In-house model list classification

The embedding-pipeline classification of the in-house model catalogue is
mirrored in `EmbeddingModelRegistry` (module `directml-config`, package
`com.aresstack.windirectml.config.models`). The registry lives in a
Java-8-compatible module so the Java-21 sidecar, the Java-8 workbench
and the Java-8 client can all reuse the same `modelId → useCase /
status / embedFamily` classification without duplicating metadata.

* **Sidecar `embed` gate** – `DirectMlPhi3Sidecar.embedFamily(...)`
  resolves `-Dembed.model` through the registry: full model IDs map
  to their `embedFamily` (`minilm` / `e5`); decoder / summarizer IDs
  are rejected with an explicit *"… is not an embedding model …"*
  error; `planned` embedding IDs that have no runtime support yet
  (Jina v2, multilingual-E5) are rejected with a status-aware message
  that points at this file.
* **Workbench embedding selector** – the `embed.model` dropdown is
  populated from `EmbeddingModelRegistry.entriesByUseCase(EMBEDDING)`,
  so only embedding entries are selectable; decoder / summarizer IDs
  are filtered out at the UI layer in addition to being rejected by
  the sidecar gate.

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

### 1.1.1 `intfloat/multilingual-e5-large-instruct` — support analysis

This subsection records the analysis behind keeping
`intfloat/multilingual-e5-large-instruct` at `status=planned`. The model
is **visible** in the Workbench `embed.model` dropdown (it has
`useCase=EMBEDDING` in the registry) and the SUPPORTED_MODELS table
above marks it 🚧 planned with `DirectML ❌ / CPU ❌`, so it cannot be
mistaken for a shipped option. Selecting it on `-Dembed.model` is
rejected by the sidecar with the `status=planned` message shown above,
which points back at this file.

**Tokenizer.** SentencePiece BPE (`sentencepiece.bpe.model`, packaged
inside Hugging Face's `tokenizer.json`). Uses `▁` as the word-boundary
marker and the XLM-R special tokens `<s>`, `</s>`, `<pad>`, `<unk>`,
`<mask>`. This is **incompatible** with the current WordPiece-only
tokenizer used by the MiniLM / E5-WordPiece paths.

**Architecture.** XLM-RoBERTa-large encoder (`config.model_type =
"xlm-roberta"`), 24 layers, hidden size 1024, 16 attention heads,
intermediate size 4096, learned absolute positional embeddings of length
514 (RoBERTa convention: position ids start at `padding_idx + 1 = 2`).
Not a drop-in for the existing BERT/E5-WordPiece core because of the
RoBERTa-style positional embedding offset and the SentencePiece input
pipeline.

**Config fields that diverge from BERT/E5.**

- `model_type = "xlm-roberta"` (not `"bert"`).
- `type_vocab_size = 1` — only a single segment embedding row, so the
  current BERT token-type-embedding code path cannot be reused without
  guarding against a 1-row table.
- `pad_token_id = 1`, `bos_token_id = 0`, `eos_token_id = 2` (RoBERTa
  convention), not the BERT `[CLS]/[SEP]` IDs.
- `max_position_embeddings = 514`, but effective max sequence length is
  512 because of the `padding_idx + 1` offset.

**Weight names.** Top-level prefix is `roberta.*` (e.g.
`roberta.embeddings.word_embeddings.weight`,
`roberta.encoder.layer.{i}.attention.self.query.weight`), not the
`bert.*` prefix the existing encoder loader expects. A loader for this
model must either accept a `roberta`-rooted state dict or be re-keyed
on the fly.

**Token-type embeddings.** The model ships with
`type_vocab_size = 1`, so the existing assumption in the BERT/E5 core
that `token_type_embeddings` has at least two rows does not hold; the
DirectML path needs an explicit single-segment branch (or the segment
embedding can be folded into the word embedding at load time).

**Pooling.** Mean pooling over `attention_mask`, identical to the
shipped MiniLM / E5-WordPiece pipeline. The existing
`MeanPoolingKernel` is reusable as soon as the encoder produces the
`[N, hidden=1024]` token-state matrix.

**Normalization.** L2-normalisation of the pooled vector, identical to
the shipped E5 path. The existing `L2NormalizeKernel` is reusable.

**Prefix / instruction format.** The `-instruct` variant expects an
instruction-style query prefix, **not** the plain `"query: "` /
`"passage: "` E5 prefix:

```text
embed_query   = "Instruct: {task}\nQuery: {query}"
embed_passage = "{passage}"   // passages are embedded without a prefix
```

`{task}` is a free-form natural-language description of the retrieval
task (e.g. *"Given a web search query, retrieve relevant passages that
answer the query"*). The existing `EmbeddingRequest.prefix` slot can
carry the full `"Instruct: …\nQuery: "` string for queries, but the
sidecar must not silently re-use the `"query: " / "passage: "` prefixes
of the WordPiece-E5 path because they would produce off-distribution
embeddings.

**Compatibility decision.** Stays `planned`. The model needs **all** of
the following before its status may be raised:

1. A SentencePiece-BPE tokenizer (shared with future XLM-R / Llama
   work).
2. An XLM-RoBERTa-aware encoder path (RoBERTa positional offset,
   `roberta.*` weight naming, `type_vocab_size = 1` handling).
3. Explicit instruction-prefix handling on the sidecar / client side so
   callers never accidentally embed with the wrong prefix.
4. A CPU real-model test before `status=experimental`, **and** a
   DirectML real-model test (or an explicit *CPU-only* marker) before
   `status=shipped`.

Until items 1–4 land, the registry entry, the SUPPORTED_MODELS table
and the sidecar gate are the single source of truth: visible in the
Workbench, advertised as planned, rejected at runtime.


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
