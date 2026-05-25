# win-directml-java

Local **embeddings and reranking** for Java applications that want to reduce
or replace cloud inference calls. A Java 8 host process talks JSON-RPC over
stdin/stdout to a Java 21 sidecar that runs MiniLM, E5 and cross-encoder
reranker workloads locally on CPU or, on Windows, with DirectML acceleration.

The supported release use case is a local RAG / search pipeline:

```text
Java 8 application
  -> chunk documents
  -> create local embeddings
  -> store/search vectors in your own vector index
  -> rerank top-K candidates locally
  -> keep sensitive text on-prem/offline
```

Use this when you want:

- **Data control:** document text and search queries do not have to leave the
  machine or network boundary.
- **Predictable cost:** embedding and reranking calls are local compute instead
  of per-token/per-request cloud billing.
- **Offline / on-prem operation:** CPU-only operation is supported; DirectML is
  an acceleration path, not a requirement.
- **Java 8 integration:** the application uses a small Java 8 client artifact;
  DirectML, Java FFM and model loading stay isolated in the Java 21 sidecar.

Decoder LLMs are intentionally not the first release focus. The existing Phi-3
summarizer path is experimental and sidecar-only. The primary supported product
path is **local embeddings + local reranking**.

## Recommended model choices

| Need | Recommended path | Notes |
|---|---|---|
| Fast default embeddings | `sentence-transformers/all-MiniLM-L6-v2` via `embed.model=minilm` | Small, quick, good release baseline. |
| Higher-quality E5-style retrieval | `intfloat/e5-base-v2` via `embed.model=e5`, `e5.model=base-v2` | Use `query: ` for queries and `passage: ` for indexed documents. |
| Maximum E5 capacity in current WordPiece path | `intfloat/e5-large-v2` | Experimental and much larger. |
| Better final ordering after vector search | `cross-encoder/ms-marco-MiniLM-L-6-v2` via `rerank` | Scores query/document pairs jointly; best used on top-K candidates. |
| German E5 STS checkpoint | `danielheinz/e5-base-sts-en-de` | Planned only: current upstream is XLM-R/SentencePiece, not the shipped WordPiece E5 path. |

See [`SUPPORTED_MODELS.md`](SUPPORTED_MODELS.md) for the full support matrix
and [`MODEL_LICENSES.md`](MODEL_LICENSES.md) for model license notes.

## CPU vs. DirectML guidance

`cpu`, `directml` and `auto` are first-class backend choices.

| Mode | When to use it | Behavior |
|---|---|---|
| `-Dembed.backend=cpu` / `-Drerank.backend=cpu` | Servers, CI, VMs or desktops without a DirectML-capable GPU | Pure local CPU path. Not a failure mode. |
| `-Dembed.backend=directml` / `-Drerank.backend=directml` | You require GPU acceleration and want failures to be visible | Fails fast if DirectML/model loading is unavailable. |
| `-Dembed.backend=auto` / `-Drerank.backend=auto` | Default application mode | Tries DirectML first and falls back to CPU, reporting the reason in health/status. |

DirectML improves latency and throughput on Windows machines with suitable GPUs.
CPU remains useful for small workloads, tests, offline environments and systems
where GPU acceleration is not available. See [`BENCHMARK.md`](BENCHMARK.md)
for reproducible throughput numbers and [`docs/fallback-policy.md`](docs/fallback-policy.md)
for the full backend contract.

## Typical local RAG pipeline

1. **Chunk documents** in the host application.
2. **Embed chunks locally** with MiniLM or E5.
3. **Store vectors** in your own vector index or database.
4. **Search top-K** by vector similarity.
5. **Rerank top-K locally** with the cross-encoder reranker.
6. Optionally pass the final context to a later local or remote summarization
   layer. Decoder LLM support is future/experimental, not the core release path.

For E5, prefix query texts with `query: ` and indexed/passages with `passage: `.
Reranker scores are raw model logits: compare them only within the same query,
not across different queries or models. The `rerank` result is sorted descending
by score.

## Architecture

### Direct Java 21 API (no sidecar)

Java 21 applications can use embeddings and reranking directly in-process,
without starting the JSON-RPC sidecar:

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

The direct API hides sidecar transport details and does not require JSON-RPC or a
separate process. Supported WordPiece E5 variants: `small-v2`, `base-v2`, `large-v2`.
XLM-R/SentencePiece E5 models (e.g. `danielheinz/e5-base-sts-en-de`) remain planned
but not ready; attempting to load them throws `UnsupportedModelException`.

### Sidecar architecture (Java 8 bridge/adapter)

The sidecar is a thin JSON-RPC adapter over the `directml-runtime` API.
It allows Java 8 host applications to access local ML capabilities without
requiring Java 21 in the host process.

```text
Java 8 host application
    │ JSON-RPC 2.0, one message per line
    │ stdin / stdout (logs on stderr)
    ▼
Java 21 DirectML sidecar (bridge/adapter over directml-runtime)
    │ Java FFM preview API
    ▼
Windows 11 DirectML / D3D12 / DXGI
```

This repository is a cleaned extraction of the DirectML and inference-relevant
code from `aresstack/win-acp-java`. ACP, MCP, graph routing and agent-host
layers are intentionally **not** included.

### Non-goals

- No ACP or MCP runtime.
- No ONNX Runtime dependency.
- No JNA or JNI dependency.
- No bundled model weights in Maven artifacts.
- No generic ONNX/GGUF/Transformers runner in the first release.
- No promise that decoder LLMs are production-ready in the core release.

## Modules

```text
directml-config                     Minimal inference configuration
directml-windows-bindings           Java 21 FFM bindings for DXGI, D3D12, DirectML, COM/HRESULT
directml-encoder                    MiniLM, E5 and reranker encoder/runtime code
directml-runtime                    Public Java 21 facade for direct in-process use (no sidecar)
directml-sidecar-protocol-java8     Shared protocol/validation types (Java 8 compatible)
directml-sidecar                    JSON-RPC 2.0 sidecar – Java 8 bridge/adapter over directml-runtime
directml-sidecar-client-java8       Pure Java 8 client for host applications
directml-sidecar-workbench          Java 8 Swing workbench / diagnostics UI
directml-inference                  Experimental Phi-3 summarizer path
```

## Requirements

- Windows 11 for DirectML acceleration.
- JDK 21 for the sidecar/runtime modules.
- Java 8 or newer for the host application using `directml-sidecar-client-java8`.
- Downloaded model directories; model weights are not shipped in Maven artifacts.

FFM is still a preview feature, so Java 21 sidecar/runtime execution uses:

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

## Run the sidecar

```powershell
./gradlew.bat :directml-sidecar:run
```

Minimal JSON-RPC traffic:

```json
{"jsonrpc":"2.0","id":"h1","method":"health","params":{}}
{"jsonrpc":"2.0","id":"x","method":"shutdown","params":{}}
```

Protocol details: [`directml-sidecar/PROTOCOL.md`](directml-sidecar/PROTOCOL.md).

### MiniLM default

```powershell
./gradlew.bat :directml-sidecar:run -Dembed.backend=auto
```

### E5 base-v2

```powershell
./gradlew.bat :directml-sidecar:run `
    -Dembed.model=e5 `
    -De5.model=base-v2 `
    -De5.modelDir=model/e5-base-v2 `
    -Dembed.backend=auto
```

### CPU-only

```powershell
./gradlew.bat :directml-sidecar:run `
    -Dembed.backend=cpu `
    -Drerank.backend=cpu
```

CPU-only mode is valid for local/offline use and CI. It should not be treated
as an error just because DirectML is unavailable.

## Java 8 host integration

The host application uses `directml-sidecar-client-java8` and does not need FFM,
DirectML bindings or Java 21 APIs on its own classpath.

```gradle
dependencies {
    implementation 'com.aresstack:directml-sidecar-client-java8:0.1.0-beta.1'
}
```

Typical host flow:

```java
SidecarClientConfig config = new SidecarClientConfig();
config.setSidecarJarPath("directml-sidecar.jar");
config.setEmbedBackend("auto");

SidecarClient client = new SidecarClient(config);
client.start();
HealthResult health = client.health();
EmbeddingResult vector = client.embed("query: local search example");
RerankResult reranked = client.rerank("local DirectML search", documents, 10);
client.shutdown();
```

The Java 8 client exposes typed result objects, JSON-RPC errors, timeout
handling and orderly process shutdown. Full copy-paste examples are planned in
`examples/java8-client`.

## Workbench

For interactive testing of the sidecar from a Java 8 host environment
(stdin/stdout JSON-RPC, no FFM and no DirectML in the host JVM):

```powershell
./gradlew.bat :directml-sidecar:jar
./gradlew.bat :directml-sidecar-workbench:run
```

Documentation: [`WORKBENCH.md`](WORKBENCH.md).

## Embeddings and reranking details

### Embeddings (`embed`, `embedBatch`)

`embed` and `embedBatch` are backed by a generic BERT-style encoder runtime.
The current shipped/experimental WordPiece families are MiniLM and E5.

- `minilm`: `sentence-transformers/all-MiniLM-L6-v2`, 384-dimensional output.
- `e5`: `intfloat/e5-small-v2`, `intfloat/e5-base-v2`, experimental
  `intfloat/e5-large-v2`, with 384/768/1024-dimensional output.

`embedBatch` preserves input order. Planned XLM-R/SentencePiece models remain
visible in `SUPPORTED_MODELS.md` but are not silently treated as runnable.

### Reranking (`rerank`)

`rerank` scores `(query, document)` pairs jointly through a cross-encoder and
returns the top-N documents sorted by raw classifier logit. It is intended for
reranking a vector-search candidate set, not for indexing the entire corpus.

Configuration:

```text
-Drerank.modelDir=<path>
-Drerank.backend=auto|directml|cpu
```

The default model location is `model/cross-encoder-ms-marco-MiniLM-L-6-v2/`.
If no reranker model is present, the sidecar still starts and `rerank` reports
`-32005 Not implemented`.

## Status summary

| Area | Status |
|---|---|
| JSON-RPC protocol | Formalized in `directml-sidecar/PROTOCOL.md`. |
| MiniLM embeddings | CPU and DirectML paths implemented. |
| E5 WordPiece embeddings | `small-v2`, `base-v2`, experimental `large-v2` runtime path. |
| Reranker | Cross-encoder reranker exposed over JSON-RPC. |
| CPU-only usage | Supported; not a failure mode. |
| DirectML usage | Supported Windows acceleration path. |
| Phi-3 summarizer | Experimental, sidecar-only, not core release focus. |
| Decoder LLMs | Future work. |

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

Published artifacts under `com.aresstack:`:

```gradle
dependencies {
    implementation 'com.aresstack:directml-config:0.1.0-beta.1'
    implementation 'com.aresstack:directml-windows-bindings:0.1.0-beta.1'
    implementation 'com.aresstack:directml-encoder:0.1.0-beta.1'
    implementation 'com.aresstack:directml-sidecar-client-java8:0.1.0-beta.1'
}
```

Module overview:

| Coordinate | Java | Purpose |
|---|---|---|
| `com.aresstack:directml-config` | 21 | Backend / feature-level / GPU-adapter configuration types. |
| `com.aresstack:directml-windows-bindings` | 21 preview | FFM bindings to D3D12 / DXGI / DirectML plus low-level kernel primitives. |
| `com.aresstack:directml-encoder` | 21 preview | MiniLM, E5 and cross-encoder reranker pipelines with CPU + DirectML parity. |
| `com.aresstack:directml-sidecar-client-java8` | 8 | JSON-RPC client to talk to the sidecar from a Java 8 host JVM. |

The `directml-sidecar` and `directml-sidecar-workbench` applications are not
published as normal Maven library dependencies; they are intended to be shipped
as self-contained distribution zips via GitHub Releases.

### Release command

```powershell
./gradlew :directml-windows-bindings:test :directml-encoder:test `
          :directml-sidecar-client-java8:test :directml-sidecar:test
.\release.ps1 0.1.0-beta.1
```

The release workflow publishes with the Central Portal configuration in the
root Gradle build and attaches sidecar/workbench distribution zips to the
matching GitHub release.
