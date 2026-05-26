# Direct Java 21 Mode

The `directml-runtime` module provides a public Java 21 API for direct
in-process use of the local ML runtime. Java 21 applications can call
embeddings, batch embeddings, and reranking **without** starting the JSON-RPC
sidecar process.

## Maven coordinates

```xml
<dependency>
    <groupId>com.aresstack</groupId>
    <artifactId>directml-runtime</artifactId>
    <version>${win-directml-java.version}</version>
</dependency>
```

Gradle:
```groovy
implementation 'com.aresstack:directml-runtime:<version>'
```

## Quick start

```java
import com.aresstack.windirectml.runtime.api.*;
import java.nio.file.Path;
import java.util.List;

// 1. Create a runtime
MlRuntime runtime = MlRuntime.create();

// 2. Load and use an embedding model (MiniLM)
var cfg = EmbeddingConfig.builder()
        .model(EmbeddingModelId.MINILM_L6_V2)
        .modelDir(Path.of("model/all-MiniLM-L6-v2"))
        .build();
try (EmbeddingModelHandle embeddings = runtime.loadEmbeddings(cfg)) {
    float[] vector = embeddings.embed("hello world");
    List<float[]> batch = embeddings.embedBatch(List.of("text one", "text two"));
}

// 2b. E5 model (default prefix applied automatically from EmbeddingModelId)
var e5Cfg = EmbeddingConfig.builder()
        .model(EmbeddingModelId.E5_BASE_V2)
        .modelDir(Path.of("model/e5-base-v2"))
        .build();
try (EmbeddingModelHandle embeddings = runtime.loadEmbeddings(e5Cfg)) {
    float[] vector = embeddings.embed("search query");
}

// 3. Load and use a reranker (modelDir derived automatically from RerankerModelId)
var rerankCfg = RerankerConfig.builder()
        .model(RerankerModelId.MS_MARCO_MINILM_L6)
        .build();
try (RerankerModelHandle reranker = runtime.loadReranker(rerankCfg)) {
    var results = reranker.rerank("search query", List.of("doc1", "doc2"));
    for (var r : results) {
        System.out.printf("index=%d score=%.4f%n", r.originalIndex(), r.score());
    }
}
```

## Backend selection

The `Backend` enum controls hardware execution:

| Backend | Description |
|---------|-------------|
| `Backend.AUTO` (default) | Try DirectML first, fall back to CPU if unavailable. |
| `Backend.DIRECTML` | Require DirectML; fail if unavailable. |
| `Backend.CPU` | Always use the pure-Java CPU backend. |

```java
MlRuntime runtime = MlRuntime.builder()
        .backend(Backend.CPU)
        .build();
```

## Supported model families

| Family | Model ID enum | Status |
|--------|--------------|--------|
| `minilm` | `EmbeddingModelId.MINILM_L6_V2` | Shipped (WordPiece, BERT-style) |
| `e5` | `EmbeddingModelId.E5_SMALL_V2`, `E5_BASE_V2`, `E5_LARGE_V2` | Shipped (WordPiece E5 variants only) |

### Supported E5 variants (WordPiece)

| Variant | Enum constant | Dimensions |
|---------|---------------|------------|
| `intfloat/e5-small-v2` | `EmbeddingModelId.E5_SMALL_V2` | 384 |
| `intfloat/e5-base-v2` | `EmbeddingModelId.E5_BASE_V2` | 768 |
| `intfloat/e5-large-v2` | `EmbeddingModelId.E5_LARGE_V2` | 1024 |

### Not yet supported (planned)

XLM-R/SentencePiece-based E5 models are **planned but not ready**:
- `danielheinz/e5-base-sts-en-de` – XLM-RoBERTa encoder, requires SentencePiece tokenizer
- `intfloat/multilingual-e5-large-instruct` – XLM-RoBERTa-large, requires SentencePiece + XLM-R encoder path

These models use a fundamentally different tokenizer (SentencePiece) and model
architecture (XLM-RoBERTa) that is not implemented in this runtime.

## Error handling

- `EmbeddingException` – checked exception for model loading or inference
  failures (missing files, corrupted weights, backend errors).
- `RerankException` – checked exception for reranker inference failures.
- `IllegalArgumentException` – thrown at config build time when `modelDir`
  does not match the expected directory name for the chosen `RerankerModelId`.

## Thread safety

`MlRuntime` is stateless and safe to share across threads. The model
handles (`EmbeddingModelHandle`, `RerankerModelHandle`) inherit thread safety
from the underlying encoder/reranker implementations (the shipped CPU and
DirectML backends are thread-safe).

## Migration from `runtime.facade`

The lower-level `com.aresstack.windirectml.runtime.facade.LocalMlRuntime` is
still functional but has been deprecated. Migrate as follows:

| Facade API | New API (`runtime.api`) |
|-----------|------------------------|
| `LocalMlRuntime.create()` | `MlRuntime.create()` |
| `LocalMlRuntime.create(LocalMlRuntimeConfig)` | `MlRuntime.builder().backend(...).build()` |
| `runtime.loadEmbeddingModel(EmbeddingModelConfig)` | `runtime.loadEmbeddings(EmbeddingConfig)` |
| `runtime.loadRerankerModel(RerankerModelConfig)` | `runtime.loadReranker(RerankerConfig)` |
| `LocalEmbeddingModel` handle | `EmbeddingModelHandle` |
| `LocalRerankerModel` handle | `RerankerModelHandle` |

## Differences from the sidecar

| Concern | Sidecar mode | Direct mode |
|---------|-------------|-------------|
| Transport | JSON-RPC over stdin/stdout | Direct method calls |
| Java version | Host: Java 8+, Sidecar: Java 21 | Java 21 only |
| Process model | Separate process | Same JVM |
| Dependencies | `directml-sidecar-client-java8` | `directml-runtime` |

The direct API is a lighter-weight integration path when the host application
already runs on Java 21 and does not need process isolation.

## Example

See [`examples/java21-direct-api/`](../examples/java21-direct-api/) for a
complete runnable example.
