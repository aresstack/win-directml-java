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

| Variant                                   | Default | DirectML | CPU reference | Tokenizer               | Pooling                                                 | Status          | Module                                                           |
|-------------------------------------------|---------|----------|---------------|-------------------------|---------------------------------------------------------|-----------------|------------------------------------------------------------------|
| `sentence-transformers/all-MiniLM-L6-v2`  | yes     | ✅        | ✅             | WordPiece (`vocab.txt`) | mean + optional L2                                      | ✅ shipped       | `directml-encoder` (`DirectMlMiniLmEncoder`, `CpuMiniLmEncoder`) |
| `intfloat/e5-small-v2`                    | no      | ✅        | ✅             | WordPiece (`vocab.txt`) | mean + L2 + `query: `/`passage: ` prefix                | ✅ shipped       | `directml-encoder` (`E5Encoders.SMALL_V2`)                       |
| `intfloat/e5-base-v2`                     | no      | ✅        | ✅             | WordPiece (`vocab.txt`) | mean + L2 + `query: `/`passage: ` prefix                | ✅ shipped       | `directml-encoder` (`E5Encoders.BASE_V2`)                        |
| `danielheinz/e5-base-sts-en-de`           | no      | ❌        | ❌             | SentencePiece (XLM-R)   | mean + L2 (E5 prefixes when supported)                  | 🚧 planned      | – (XLM-R/SentencePiece pending; see §1.0.1)                      |
| `intfloat/e5-large-v2`                    | no      | ✅        | ✅             | WordPiece (`vocab.txt`) | mean + L2 + `query: `/`passage: ` prefix                | 🧪 experimental | `directml-encoder` (`E5Encoders.LARGE_V2`)                       |
| `intfloat/multilingual-e5-large-instruct` | no      | ❌        | ❌             | SentencePiece (XLM-R)   | mean + L2 + `Instruct: …\nQuery: …` prefix              | 🚧 planned      | – (needs SentencePiece + XLM-RoBERTa core)                       |
| `jinaai/jina-embeddings-v2-base-de`       | no      | ❌        | ❌             | WordPiece (Jina-custom) | mean + L2; ALiBi positional bias, max 8192 tokens       | 🚧 planned      | – (needs custom Jina v2 attention path; see §1.1.2)              |
| `nomic-ai/nomic-embed-text-v1.5`          | no      | ❌        | ❌             | WordPiece               | mean + L2 + `search_query:` / `search_document:` prefix | 🚧 planned      | –                                                                |

All shipped embedding models go through the same
`com.aresstack.windirectml.encoder.EmbeddingModel` interface and the
batched `embedBatch(...)` path with bucket-padded sequence lengths.
DirectML pooling and L2 normalisation are GPU-resident; only the final
`[N, hidden]` matrix is read back to the host.

For reproducible throughput measurements (`embed` and `embedBatch`,
CPU + DirectML, `N ∈ {10, 50, 100}`) of the shipped embedding models
and the derived backend / model recommendations, see
[`BENCHMARK.md`](BENCHMARK.md).

### 1.0.1 MiniLM / E5 runtime selection and validation

- **MiniLM (`sentence-transformers/all-MiniLM-L6-v2`)**
    - Registry status: `shipped` (`embedFamily=minilm`, CPU + DirectML).
    - `-Dembed.model` accepts both alias `minilm` and full model ID
      `sentence-transformers/all-MiniLM-L6-v2`.
    - Works through both `embed` and `embedBatch`.
- **E5 (`danielheinz/e5-base-sts-en-de`)**
    - Registry status: `planned` (no `embedFamily` while the runtime is
      WordPiece-only).
    - The upstream checkpoint at
      https://huggingface.co/danielheinz/e5-base-sts-en-de hosts an
      `XLMRobertaModel` (`vocab_size=250002`, `type_vocab_size=1`,
      SentencePiece tokenizer, ~1.06 GB `model.safetensors`), i.e. a
      multilingual-E5 derivative – not the WordPiece BERT-base profile
      the current E5 runtime supports.
    - `-Dembed.model=danielheinz/e5-base-sts-en-de` is rejected by the
      `embed` gate with the standard
      `… classified as an embedding model but has no runtime support in
      this build (status=planned).` message until SentencePiece + XLM-R
      support lands (tracked with multilingual-E5).
    - The variant constant `E5Encoders.BASE_STS_EN_DE` and the
      `download-e5.ps1 -Variant base-sts-en-de` helper are kept in the
      tree but are not exercised end-to-end against the current upstream
      checkpoint.
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

| `modelId`                                 | `useCase`  | `status`        | Backend support   | Notes                                                                                                                                  |
|-------------------------------------------|------------|-----------------|-------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| `sentence-transformers/all-MiniLM-L6-v2`  | embedding  | ✅ shipped       | CPU + DirectML    | Default fast embedding model (WordPiece, BERT-style).                                                                                  |
| `danielheinz/e5-base-sts-en-de`           | embedding  | 🚧 planned      | – (planned)       | Upstream checkpoint is an XLMRobertaModel (vocab=250002, type_vocab=1); needs SentencePiece + XLM-R, tracked with multilingual-E5.     |
| `intfloat/multilingual-e5-large-instruct` | embedding  | 🚧 planned      | – (planned)       | NOT compatible with the current WordPiece-only E5 path. Requires SentencePiece + XLM-RoBERTa core.                                     |
| `jinaai/jina-embeddings-v2-base-de`       | embedding  | 🚧 planned      | – (planned)       | Jina BERT v2 uses ALiBi positional bias + GLU-style MLP; not a drop-in for the current BERT/MiniLM/E5 core. Stays planned, see §1.1.2. |
| `openai/gpt-oss-120b`                     | decoder    | ⛔ unsupported   | – (not for embed) | Decoder-only LLM. Rejected by the `embed` endpoint.                                                                                    |
| `casperhansen/llama-3.3-70b-instruct-awq` | decoder    | ⛔ unsupported   | – (not for embed) | Llama 3.3 70B AWQ-quantised decoder-only LLM. Rejected by the `embed` endpoint.                                                        |
| `ellamind/summarizer-v6-llama-v2`         | summarizer | ⛔ unsupported   | – (not for embed) | Llama-v2 summarizer fine-tune. Belongs to a future text-generation/summarize ticket, not the embed endpoint.                           |
| `microsoft/Phi-3-mini-4k-instruct-onnx`   | summarizer | 🧪 experimental | CPU (Workbench, from wdmlpack) + DirectML (sidecar) | Native Java/DirectML decoder (no Python/ONNX Runtime). Workbench: Download → Convert → `model_phi3.wdmlpack`, then the Summarizer runs it heap-light via `Phi3RuntimePackage` → `Phi3Runtime` (CPU). Also runs via the sidecar `summarize` path. Runnable since PHI3-RUNTIME-HEAPLIGHT-1. |
| `microsoft/Phi-3.5-mini-instruct-onnx`    | summarizer | 🚧 planned      | – (planned)       | Phi-3 successor; same architecture. Not executable in the Workbench (no wdmlpack compiler).                                            |

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

### 1.1.2 `jinaai/jina-embeddings-v2-base-de` — analysis and status decision

This subsection records the architectural analysis behind the `planned`
status of `jinaai/jina-embeddings-v2-base-de` in the registry. It is
the source-of-truth referenced from the registry entry's `notes` field
and from the unimplemented-embedding error message.

**Source of truth.** The analysis is based on the public model card
and `config.json` of
[`jinaai/jina-embeddings-v2-base-de`](https://huggingface.co/jinaai/jina-embeddings-v2-base-de)
(JinaBERT v2 architecture, custom modelling code released under
`trust_remote_code=True`).

| Property             | Value                                                                                                                                                                                             |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Architecture family  | JinaBERT v2 (BERT-base sized encoder; 12 layers / 768 hidden / 12 attention heads)                                                                                                                |
| Tokenizer            | WordPiece (`tokenizer.json`, Jina-custom bilingual de/en vocab; not byte-identical to standard BERT)                                                                                              |
| Max sequence length  | 8192 tokens (enabled by ALiBi extrapolation)                                                                                                                                                      |
| Positional encoding  | **ALiBi** — Attention with Linear Biases; no learned `position_embeddings` weight in the checkpoint                                                                                               |
| Feed-forward block   | **GLU-style** MLP (gated linear unit) — differs from the single-projection + GELU used by BERT/MiniLM/E5                                                                                          |
| Pooling              | Mean pooling over `last_hidden_state` with attention-mask weighting                                                                                                                               |
| Normalisation        | L2-normalised output embeddings (cosine similarity is the supported metric)                                                                                                                       |
| Output dimension     | 768                                                                                                                                                                                               |
| Custom modeling code | Yes (`trust_remote_code=True`; the checkpoint ships its own `modeling_bert.py`)                                                                                                                   |
| Weight naming        | BERT-prefixed (`bert.encoder.layer.*`) but **without** `bert.embeddings.position_embeddings.weight`, **with** per-layer ALiBi slope tensors and **with** an extra MLP projection for the GLU gate |

**Compatibility with the current BERT / MiniLM / E5 core**

The current shipped encoder stack (`directml-encoder`'s
`DirectMlBertEncoderLayerBlock` / `BertEncoderConfig` and the MiniLM /
E5 runtimes) assumes:

1. **Learned absolute positional embeddings** added at the embedding
   layer (`bert.embeddings.position_embeddings`). Jina v2 has no such
   weight; positions are injected as a per-head bias inside every
   attention layer.
2. **Vanilla BERT feed-forward** (`intermediate.dense` → GELU →
   `output.dense`). Jina v2 uses a GLU variant, which roughly doubles
   the intermediate parameter count and changes the wiring of the FFN.
3. **Max sequence length ≤ 512** in the runtime bucket configuration.
   Jina v2 advertises 8192 tokens; supporting that on DirectML would
   require new bucket sizes and a memory-budget review.

Because of (1)+(2), `jinaai/jina-embeddings-v2-base-de` is **not** a
drop-in for the existing BERT core, neither for MiniLM nor for E5.

**Implementation decision**

- Status: **`planned`** (no runtime support in this build).
- Adding Jina v2 requires either
    - a Jina-specific attention path (ALiBi bias addition) **and**
      a GLU feed-forward block on top of the existing
      `DirectMlBertEncoderLayerBlock`, **or**
    - a CPU-only starter path (Java reference implementation) before any
      DirectML work, gated behind a real-model test.
- Until at least the CPU-only path exists with a real-model reference
  test, the status must not be promoted to `experimental`. Promotion
  to `shipped` requires CPU + DirectML parity, matching the contract
  documented for MiniLM and E5 above.

**Workbench behaviour**

The Workbench `embed.model` dropdown is populated from
`EmbeddingModelRegistry.entriesByUseCase(EMBEDDING)`, so
`jinaai/jina-embeddings-v2-base-de` is visible alongside the shipped
embedding model IDs. It is not silently treated as `shipped`: the
sidecar's `embed` gate rejects it at startup with the status-aware
message shown above (`status=planned`, pointer to this file), so
selecting it from the dropdown produces an explicit error rather than
a partial-result success.

**Download script**

`scripts/download-jina.ps1` is intentionally **not added in this PR**.
The download helpers (`download-minilm.ps1`, `download-e5.ps1`,
`download-reranker.ps1`) exist to auto-enable real-model reference
tests once weights are present locally; there is no Jina v2 runtime
yet, so a download script would only fetch weights for code that
cannot consume them. The script will be introduced together with the
first CPU runtime path (the directory hints
`model/jina-embeddings-v2-base-de` and
`model/jinaai/jina-embeddings-v2-base-de` are reserved in the registry
so the future script and runtime agree on the layout).

**Required artefacts (for a future `download-jina.ps1`)**

For reference, a future download script will need at least the
following files from the HuggingFace repo (paths relative to the
checkpoint root):

- `model.safetensors` (or `pytorch_model.bin`)
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `vocab.txt`
- `modules.json` and `1_Pooling/config.json` (sentence-transformers
  metadata, optional but useful to assert mean pooling)

Missing-file errors must point at the specific artefact (mirroring
the wording the Phi-3 loader uses: *"Jina v2 model directory is
missing tokenizer.json"*) once the loader exists.

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
| `microsoft/Phi-3.5-mini-instruct-onnx`                        | TBD             | –        | –             | –        | 🚧 planned      | –                    |
| `Qwen2.5-Coder-0.5B-Instruct`                                 | INT4 AWQ b128   | ✅        | ✅             | paged    | 🧪 experimental | `directml-inference` |
| `Qwen2.5-Coder-1.5B-Instruct` (ONNX source TBD/research)      | INT4 AWQ b128   | –        | –             | –        | 🚧 planned      | TBD/planned          |
| `Qwen2.5-Coder-3B-Instruct` (ONNX source TBD/research)        | INT4 AWQ b128   | –        | –             | –        | 🚧 planned      | TBD/planned          |

The Phi-3 pipeline runs prefill and decode in a single DirectML graph
per layer; speculative decoding, batched generation and beam search are
out of scope for the current release. The Phi-3 model weights are not
permissively licensed — read
https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx
before redistributing anything you build on top of them.

Qwen2.5-Coder is still planned/not runnable in this project; the ONNX source is
explicitly TBD/research until a resolvable and layout-compatible repository is
verified. Scale-up candidates (1.5B, 3B) use the same artifact format and will
not be enabled until the 0.5B runtime smoke test passes (see #99).

### 3.1 Summarization via Decoder Models

The **Workbench Summarizer tab** uses decoder models (not embeddings)
for text generation. The summarizer model selector is populated from
`EmbeddingModelRegistry.entriesByUseCase(SUMMARIZER)` and is
**independent** of the embedding model selector.

| `modelId`                               | `useCase`  | `status`        | Workbench support | Notes                                                                                                     |
|-----------------------------------------|------------|-----------------|-------------------|-----------------------------------------------------------------------------------------------------------|
| `microsoft/Phi-3-mini-4k-instruct-onnx` | summarizer | 🧪 experimental | ✅ runnable (Download → Convert → run) | Native Java/DirectML decoder (no Python/ONNX Runtime), ~2.3 GB INT4. Workbench Convert produces `model_phi3.wdmlpack` (`Phi3PackageLifecycle`) and the Summarizer runs it heap-light (`Phi3RuntimePackage` → `Phi3Runtime`, CPU). Runnable since PHI3-RUNTIME-HEAPLIGHT-1. |
| `microsoft/Phi-3.5-mini-instruct-onnx`  | summarizer | 🚧 planned      | ❌ not yet         | Successor; same architecture. Not executable in the Workbench (no wdmlpack compiler).                     |
| `Qwen/Qwen2.5-Coder-0.5B-Instruct`      | causal-lm  | 🧪 experimental | ✅ runnable        | Qwen2.5-Coder 0.5B. Native DirectML INT4 runtime (`model_q4f16.wdmlpack`), no Python. ChatML template. See [`docs/qwen-smoke-test.md`](docs/qwen-smoke-test.md). |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct`      | causal-lm  | 🚧 planned      | ❌ not yet         | Scale-up candidate (~1 GB INT4). Blocked on 0.5B runtime verification.                                    |
| `Qwen/Qwen2.5-Coder-3B-Instruct`        | causal-lm  | 🚧 planned      | ❌ not yet         | Scale-up candidate (~2 GB INT4). Blocked on 0.5B runtime verification.                                    |
| `ellamind/summarizer-v6-llama-v2`       | summarizer | ⛔ unsupported   | ❌                 | Llama-v2 fine-tune; no local runtime path in this project.                                                |

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

### 3.2 Workbench Summarizer smoke test (manual)

> **Phi-3-mini is runnable in the Workbench Summarizer (PHI3-RUNTIME-HEAPLIGHT-1).** Download → Convert produces
> `model_phi3.wdmlpack` (`Phi3PackageLifecycle`), and the Summarizer runs it heap-light via `Phi3RuntimePackage` →
> native `Phi3Runtime` (CPU; no Python/ONNX Runtime). Status is EXPERIMENTAL/runnable. The steps below are the
> current Workbench smoke; the historical "not executable" note further down describes the original audit state.
>
> _Historical (PHI3-PRODUCT-AUDIT-1): Phi-3 was temporarily not executable in the Workbench while it lacked a
> wdmlpack compiler. That is resolved — PHI3-WDMLPACK-COMPILER-1/2 added the compiler and PHI3-RUNTIME-HEAPLIGHT-1
> made the package load heap-light, so the smoke below applies again (now with a Convert step)._

1. Download Phi-3 from the Workbench **Download** tab (button:
   "Download Phi-3 Mini 4K Instruct (Summarizer)").
2. **Convert** the downloaded model to `model_phi3.wdmlpack` from the Download tab (the Phi-3 row's package action).
3. Switch to the **Summarizer** tab.
4. Select `microsoft/Phi-3-mini-4k-instruct-onnx` from the model dropdown.
5. Paste a longer text (≥3 sentences).
6. Click **Summarize**.
7. Verify that the output area shows generated summary text (not
   extractive sentences).

### 3.3 Qwen2.5-Coder 0.5B CPU smoke test (manual)

> Tracked in issue #101. Full smoke-test protocol in
> [`docs/qwen-smoke-test.md`](docs/qwen-smoke-test.md).

> **⚠️ Experimental / CPU-only:** The Qwen CPU runtime is not wired into
> the Workbench UI or registered as a runnable backend. Status remains
> planned/not-runnable until the ONNX source and layout are verified
> end-to-end (issue #100). The smoke test requires explicit opt-in via
> `-Dqwen.enable.experimental.runtime=true`.

1. Download Qwen2.5-Coder-0.5B-Instruct ONNX model into
   `model/qwen2.5-coder-0.5b-directml-int4/` (download script tracked in
   issue #100).
2. Run the automated smoke test:
   ```bash
   ./gradlew :directml-inference:test \
       --tests "*.qwen.QwenCpuSmokeTest" \
       -Dqwen.model.dir=model/qwen2.5-coder-0.5b-directml-int4 \
       -Dqwen.enable.experimental.runtime=true
   ```
3. Verify all four prompt scenarios produce non-empty output:
    - English summarization
    - German summarization
    - Natural/ADABAS code explanation
    - Short max-token generation (≤32 tokens)
4. Missing-file diagnostics are covered by CI-safe unit tests:
   ```bash
   ./gradlew :directml-inference:test --tests "*.qwen.QwenModelDirValidatorTest"
   ```

## 4. Sidecar / JSON-RPC

The `directml-sidecar` module exposes the embedding, reranking and
text-generation endpoints over a local JSON-RPC port (no network bind
by default). The pure-Java-8 client (`directml-sidecar-client-java8`,
also on Maven Central) is the supported way for Java-8 host
applications to call the sidecar. See
[`directml-sidecar/PROTOCOL.md`](directml-sidecar/PROTOCOL.md) for the
wire format.

## 5. Hardware requirements

| Requirement                        | Why                                                                                                                                                                                  |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Windows 10 21H1+ / Windows 11      | DirectML in-box (1.8+); side-by-side `DirectML.dll` is also supported via `-Dwindirectml.directml.dll=…`. Some optional fast paths use newer DirectML feature levels when available. |
| A DirectX-12-capable GPU           | every shipped backend dispatches via D3D12 + DirectML                                                                                                                                |
| Feature level ≥ DirectML 1.0 / 2.0 | all encoder kernels stay on the FL 1.0 / 2.0 baseline (composite GELU fallback for FL < 5.1)                                                                                         |
| ≥ 2 GB free GPU memory             | comfortable headroom for E5-base + bucketed batch padding                                                                                                                            |
| Java 21 (host)                     | code uses the FFM API                                                                                                                                                                |
| Java 8 (sidecar client only)       | the sidecar client artifact is compiled with `-release 8`                                                                                                                            |

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
- Phi-3.5 Mini Instruct ONNX summarizer support (pending official ONNX graph).
- Qwen2.5-Coder 1.5B / 3B scale-up (same ONNX INT4 format; blocked on 0.5B smoke test).
