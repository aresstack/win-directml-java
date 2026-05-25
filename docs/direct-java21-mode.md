# Direct Java 21 Mode

The `directml-runtime` module provides a public Java 21 facade for direct
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
import com.aresstack.windirectml.encoder.e5.E5Variant;
import com.aresstack.windirectml.runtime.facade.*;
import java.nio.file.Path;
import java.util.List;

// 1. Create a runtime
LocalMlRuntime runtime = LocalMlRuntime.create();

// 2. Load and use an embedding model (MiniLM)
var cfg = EmbeddingModelConfig.miniLm(Path.of("model/all-MiniLM-L6-v2"));
try (LocalEmbeddingModel embeddings = runtime.loadEmbeddingModel(cfg)) {
    float[] vector = embeddings.embed("hello world");
    List<float[]> batch = embeddings.embedBatch(List.of("text one", "text two"));
}

// 2b. E5 model (requires explicit variant selection)
var e5Cfg = EmbeddingModelConfig.e5(
        Path.of("model/e5-base-v2"), E5Variant.BASE_V2, "query: ");
try (LocalEmbeddingModel embeddings = runtime.loadEmbeddingModel(e5Cfg)) {
    float[] vector = embeddings.embed("search query");
}

// 3. Load and use a reranker
var rerankCfg = new RerankerModelConfig(
        Path.of("model/cross-encoder-ms-marco-MiniLM-L-6-v2"));
try (LocalRerankerModel reranker = runtime.loadRerankerModel(rerankCfg)) {
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
var config = LocalMlRuntimeConfig.builder()
        .backend(Backend.CPU)
        .build();
LocalMlRuntime runtime = LocalMlRuntime.create(config);
```

## Supported model families

| Family | Config factory | Status |
|--------|---------------|--------|
| `minilm` | `EmbeddingModelConfig.miniLm(dir)` | Shipped (WordPiece, BERT-style) |
| `e5` | `EmbeddingModelConfig.e5(dir, variant, prefix)` | Shipped (WordPiece E5 variants only) |

### Supported E5 variants (WordPiece)

| Variant | Enum constant | Dimensions |
|---------|---------------|------------|
| `intfloat/e5-small-v2` | `E5Variant.SMALL_V2` | 384 |
| `intfloat/e5-base-v2` | `E5Variant.BASE_V2` | 768 |
| `intfloat/e5-large-v2` | `E5Variant.LARGE_V2` | 1024 |

### Not yet supported (planned)

XLM-R/SentencePiece-based E5 models are **planned but not ready**:
- `danielheinz/e5-base-sts-en-de` – XLM-RoBERTa encoder, requires SentencePiece tokenizer
- `intfloat/multilingual-e5-large-instruct` – XLM-RoBERTa-large, requires SentencePiece + XLM-R encoder path

These models use a fundamentally different tokenizer (SentencePiece) and model
architecture (XLM-RoBERTa) that is not implemented in this runtime.
Attempting to load them will throw `UnsupportedModelException` with an explicit
message explaining why the model is not available.

## Error handling

- `EmbeddingException` – checked exception for model loading or inference
  failures (missing files, corrupted weights, backend errors).
- `UnsupportedModelException` – runtime exception thrown when an unsupported
  model family is requested. The exception carries the family name and a human-
  readable explanation.
- `RerankException` – checked exception for reranker inference failures.

## Thread safety

`LocalMlRuntime` is stateless and safe to share across threads. The model
handles (`LocalEmbeddingModel`, `LocalRerankerModel`) inherit thread safety
from the underlying encoder/reranker implementations (the shipped CPU and
DirectML backends are thread-safe).

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
