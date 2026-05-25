# win-directml-java

Local **embeddings and reranking** for Java 21 applications that want to reduce
or replace cloud inference calls. The core product is a pure Java API for a
Windows CPU / DirectML inference engine. It can be embedded directly in another
Java 21 application, such as an ACP sidecar or manager-agent project, without
starting this repository's legacy JSON-RPC sidecar process.

The supported release use case is a local RAG / search pipeline:

```text
Java 21 application / ACP sidecar / manager project
  -> directml-runtime API
  -> local embeddings
  -> your vector index
  -> local reranking
  -> sensitive text stays on-prem/offline
```

Use this when you want:

- **Data control:** document text and search queries do not have to leave the
  machine or network boundary.
- **Predictable cost:** embedding and reranking calls are local compute instead
  of per-token/per-request cloud billing.
- **Offline / on-prem operation:** CPU-only operation is supported; DirectML is
  an acceleration path, not a requirement.
- **Java 21 integration:** the inference engine is consumed as a normal Java
  library. A sidecar/container/process wrapper is optional and belongs to the
  consuming application architecture.

Decoder LLMs, ACP/MCP/A2A orchestration and manager-agent logic are intentionally
not part of this repository. Those belong in a separate ACP sidecar / manager
project that consumes this library as a local ML capability.

## Recommended model choices

| Need | Recommended path | Notes |
|---|---|---|
| Fast default embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Small, quick, good release baseline. |
| Higher-quality E5-style retrieval | `intfloat/e5-base-v2` | Use `query: ` for queries and `passage: ` for indexed documents. |
| Maximum E5 capacity in current WordPiece path | `intfloat/e5-large-v2` | Experimental and much larger. |
| Better final ordering after vector search | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Scores query/document pairs jointly; best used on top-K candidates. |
| German E5 STS checkpoint | `danielheinz/e5-base-sts-en-de` | Planned only: current upstream is XLM-R/SentencePiece, not the shipped WordPiece E5 path. |

See [`SUPPORTED_MODELS.md`](SUPPORTED_MODELS.md) for the full support matrix
and [`MODEL_LICENSES.md`](MODEL_LICENSES.md) for model license notes.

## CPU vs. DirectML guidance

`cpu`, `directml` and `auto` are first-class backend choices.

| Mode | When to use it | Behavior |
|---|---|---|
| `Backend.CPU` | Servers, CI, VMs or desktops without a DirectML-capable GPU | Pure local CPU path. Not a failure mode. |
| `Backend.DIRECTML` | You require GPU acceleration and want failures to be visible | Fails fast if DirectML/model loading is unavailable. |
| `Backend.AUTO` | Default application mode | Tries DirectML first and falls back to CPU. |

DirectML improves latency and throughput on Windows machines with suitable GPUs.
CPU remains useful for small workloads, tests, offline environments and systems
where GPU acceleration is not available. See [`BENCHMARK.md`](BENCHMARK.md)
for reproducible throughput numbers and [`docs/fallback-policy.md`](docs/fallback-policy.md)
for the backend contract.

## Typical local RAG pipeline

1. **Chunk documents** in the host application.
2. **Embed chunks locally** with MiniLM or E5.
3. **Store vectors** in your own vector index or database.
4. **Search top-K** by vector similarity.
5. **Rerank top-K locally** with the cross-encoder reranker.
6. Optionally pass the final context to a later local or remote summarization
   layer owned by your application.

For E5, prefix query texts with `query: ` and indexed/passages with `passage: `.
Reranker scores are raw model logits: compare them only within the same query,
not across different queries or models. The `rerank` result is sorted descending
by score.

## Direct Java 21 API

Java 21 applications can use embeddings and reranking directly in-process:

```java
import com.aresstack.windirectml.encoder.e5.E5Variant;
import com.aresstack.windirectml.runtime.facade.*;

// Create runtime (defaults to auto backend: DirectML → CPU fallback)
LocalMlRuntime runtime = LocalMlRuntime.create();

// Load and use embedding model (MiniLM)
var embedCfg = EmbeddingModelConfig.miniLm(Path.of("model/all-MiniLM-L6-v2"));
try (var embeddings = runtime.loadEmbeddingModel(embedCfg)) {
    float[] vector = embeddings.embed("hello world");
    List<float[]> batch = embeddings.embedBatch(List.of("a", "b", "c"));
}

// E5 requires an explicit WordPiece variant (SMALL_V2, BASE_V2, LARGE_V2)
var e5Cfg = EmbeddingModelConfig.e5(Path.of("model/e5-base-v2"), E5Variant.BASE_V2, "query: ");
try (var embeddings = runtime.loadEmbeddingModel(e5Cfg)) {
    float[] vector = embeddings.embed("search query");
}

// Load and use reranker
var rerankCfg = new RerankerModelConfig(Path.of("model/cross-encoder-ms-marco-MiniLM-L-6-v2"));
try (var reranker = runtime.loadRerankerModel(rerankCfg)) {
    var results = reranker.rerank("search query", List.of("doc1", "doc2"));
}
```

Dependency (Gradle):

```groovy
implementation 'com.aresstack:directml-runtime:<version>'
```

The API hides transport details and does not require JSON-RPC or a separate
process. Supported WordPiece E5 variants: `small-v2`, `base-v2`, `large-v2`.
XLM-R/SentencePiece E5 models (e.g. `danielheinz/e5-base-sts-en-de`) remain
planned but not ready; attempting to load them throws `UnsupportedModelException`.

## Relationship to sidecars and ACP

This repository is a Windows inference library. It deliberately does not contain
ACP, MCP, A2A, graph routing or manager-agent orchestration.

A future ACP sidecar or manager-agent project should depend on `directml-runtime`
and call the Java 21 API directly:

```text
ACP / manager project
  -> directml-runtime
  -> directml-encoder
  -> directml-windows-bindings
  -> Windows CPU / DirectML
```

The older JSON-RPC sidecar, Java 8 client, protocol module and workbench remain
in the source tree as beta-era compatibility code. They are no longer the primary
product path and are not part of new Maven Central releases unless explicitly
reactivated later.

## Modules

```text
directml-config                     Minimal inference configuration and registry types
directml-windows-bindings           Java 21 FFM bindings for DXGI, D3D12, DirectML, COM/HRESULT
directml-encoder                    MiniLM, E5 and reranker encoder/runtime code
directml-runtime                    Public Java 21 API for direct in-process use

Legacy / beta-era modules kept in source, not published in new releases:
directml-inference                  Experimental Phi-3 summarizer path
directml-sidecar-protocol-java8     Shared protocol/validation types for old sidecar bridge
directml-sidecar                    JSON-RPC 2.0 sidecar adapter over directml-runtime
directml-sidecar-client-java8       Java 8 client for the old JSON-RPC sidecar
directml-sidecar-workbench          Java 8 Swing workbench / diagnostics UI
```

## Requirements

- Windows 11 for DirectML acceleration.
- JDK 21 for the runtime modules.
- Downloaded model directories; model weights are not shipped in Maven artifacts.

FFM is still a preview feature, so Java 21 runtime execution uses:

```text
--enable-preview
--enable-native-access=ALL-UNNAMED
```

## Download local models

Fetch model files once. The scripts keep the files under `model/` by default
and run `scripts/model-doctor.ps1` after download.

```powershell
pwsh scripts/download-minilm.ps1                 # MiniLM default, 384 dimensions
pwsh scripts/download-e5.ps1                     # E5 base-v2 default, 768 dimensions
pwsh scripts/download-e5.ps1 -Variant small-v2   # E5 small, 384 dimensions
pwsh scripts/download-e5.ps1 -Variant base-v2    # E5 base, 768 dimensions
pwsh scripts/download-e5.ps1 -Variant large-v2   # E5 large, 1024 dimensions, experimental
pwsh scripts/download-reranker.ps1               # Reranker default ms-marco MiniLM L-6
pwsh scripts/download-reranker.ps1 -Variant ms-marco-MiniLM-L-12-v2
```

Common script options:

| Parameter | Meaning |
|---|---|
| `-ModelRoot` | Parent directory for model folders, default `model/`. |
| `-Variant` | Model variant to download. |
| `-Force` | Re-download even if files already exist. |
| `-Validate` | Display SHA-256 checksums and file sizes after download. |

`download-e5.ps1` intentionally does **not** offer
`danielheinz/e5-base-sts-en-de`; the current upstream checkpoint is
XLM-R/SentencePiece and is tracked as planned until the SentencePiece/XLM-R
runtime path exists.

## Embeddings and reranking details

### Embeddings

Embeddings are backed by a generic BERT-style encoder runtime. The current
shipped/experimental WordPiece families are MiniLM and E5.

- `minilm`: `sentence-transformers/all-MiniLM-L6-v2`, 384-dimensional output.
- `e5`: `intfloat/e5-small-v2`, `intfloat/e5-base-v2`, experimental
  `intfloat/e5-large-v2`, with 384/768/1024-dimensional output.

`embedBatch` preserves input order. Planned XLM-R/SentencePiece models remain
visible in `SUPPORTED_MODELS.md` but are not silently treated as runnable.

### Reranking

Reranking scores `(query, document)` pairs jointly through a cross-encoder and
returns the top-N documents sorted by raw classifier logit. It is intended for
reranking a vector-search candidate set, not for indexing the entire corpus.

The default model location in examples is
`model/cross-encoder-ms-marco-MiniLM-L-6-v2/`.

## Status summary

| Area | Status |
|---|---|
| Direct Java 21 API | Primary product path, implemented in `directml-runtime`. |
| MiniLM embeddings | CPU and DirectML paths implemented. |
| E5 WordPiece embeddings | `small-v2`, `base-v2`, experimental `large-v2` runtime path. |
| Reranker | Cross-encoder reranker available through the direct API. |
| CPU-only usage | Supported; not a failure mode. |
| DirectML usage | Supported Windows acceleration path. |
| Legacy JSON-RPC sidecar / Java 8 bridge | Source kept, no longer published in new releases. |
| Phi-3 summarizer | Legacy/experimental; not part of the core release path. |
| Decoder LLMs / ACP / MCP / A2A | Out of scope for this repository. |

## Build

PowerShell:

```powershell
$env:JAVA_HOME = "C:\Program Files\Zulu\zulu-21"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
./gradlew.bat clean build
```

See [`howtobuild.md`](howtobuild.md) for details.

## Runtime configuration

| System property | Default | Effect |
|---|---|---|
| `windirectml.debug` | `false` | Enables the D3D12 + DirectML debug layer and surfaces InfoQueue messages in exceptions. |
| `windirectml.directml.dll` | unset, use in-box `System32` | Absolute path to a deliberately shipped `DirectML.dll` redistributable. |
| `windirectml.dxgi.adapterIndex` | `0` | DXGI adapter index to bind. |

DirectML operators are gated by `DML_FEATURE_LEVEL`; unsupported fused kernels
fall back to portable composite kernels or skip tests where appropriate. Do not
point `windirectml.directml.dll` at DLLs installed by unrelated software under
`C:\Program Files\...`; use the in-box DLL or an application-owned
redistributable.

## Releases / Maven Central

New Maven Central releases publish only the core inference artifacts: the Java 21
runtime modules plus the shared `directml-config` module.

```gradle
dependencies {
    implementation 'com.aresstack:directml-runtime:<version>'
}
```

Published artifacts under `com.aresstack:`:

| Coordinate | Java | Purpose |
|---|---|---|
| `com.aresstack:directml-config` | 8/21 shared | Configuration, limits and model registry types. |
| `com.aresstack:directml-windows-bindings` | 21 preview | FFM bindings to D3D12 / DXGI / DirectML plus low-level kernel primitives. |
| `com.aresstack:directml-encoder` | 21 preview | MiniLM, E5 and cross-encoder reranker pipelines with CPU + DirectML parity. |
| `com.aresstack:directml-runtime` | 21 preview | Public direct Java 21 ML API for embeddings and reranking. |

Legacy beta artifacts such as `directml-sidecar`, `directml-sidecar-client-java8`,
`directml-sidecar-protocol-java8`, `directml-sidecar-workbench` and
`directml-inference` are kept in the repository but are not published in new
releases.

Full artifact overview: [`docs/artifact-structure.md`](docs/artifact-structure.md).

### Release command

```powershell
./gradlew :directml-windows-bindings:test :directml-encoder:test :directml-runtime:test
.\release.ps1 <version>
```

The release workflow publishes the core artifacts with the Central Portal
configuration in the root Gradle build.
