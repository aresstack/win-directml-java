# Runtime API – Direct Java 21 Mode vs Java 8 Sidecar Bridge

This project supports two integration modes:

## Mode 1: Direct Java 21 Library (recommended for Java 21+ applications)

Java 21 applications use the `directml-runtime` module directly as a library
dependency. No sidecar process, no JSON-RPC – just a plain in-process API call.

### Maven coordinates

```xml
<dependency>
    <groupId>com.aresstack</groupId>
    <artifactId>directml-runtime</artifactId>
    <version>${windirectml.version}</version>
</dependency>
```

### Usage

```java
import com.aresstack.windirectml.runtime.*;
import com.aresstack.windirectml.encoder.*;
import com.aresstack.windirectml.encoder.reranker.*;

// Build the runtime (loads model weights, initialises backends)
try (WinDirectMlRuntime runtime = WinDirectMlRuntime.builder()
        .embeddingModelDir(Path.of("model/all-MiniLM-L6-v2"))
        .rerankerModelDir(Path.of("model/cross-encoder-ms-marco-MiniLM-L-6-v2"))
        .backend(Backend.AUTO)          // AUTO | CPU | DIRECTML
        .embeddingFamily(EmbeddingFamily.MINILM)  // MINILM | E5
        .build()) {

    // Check readiness
    if (!runtime.isEmbeddingReady()) {
        System.err.println("Embedding model not available");
    }

    // Single embedding
    EmbeddingVector vec = runtime.embed(EmbeddingRequest.of("Hello world"));
    float[] values = vec.values();   // 384-dim for MiniLM

    // Batch embeddings
    List<EmbeddingVector> batch = runtime.embedBatch(List.of(
            EmbeddingRequest.of("First sentence"),
            EmbeddingRequest.of("Second sentence")
    ));

    // Reranking
    List<RerankResult> ranked = runtime.rerank(
            new RerankRequest("search query",
                    List.of("candidate doc A", "candidate doc B", "candidate doc C"),
                    2));  // topN=2
    // ranked is sorted by descending score
}
```

### Backend selection

| Value      | Behaviour                                              |
|------------|--------------------------------------------------------|
| `AUTO`     | Try DirectML (GPU); silently fall back to CPU if unavailable |
| `CPU`      | Force CPU execution; fail with `ModelReadinessException` if load fails |
| `DIRECTML` | Force DirectML; fail with `ModelReadinessException` if unavailable |

### Error handling

- **`ModelReadinessException`** (unchecked) – thrown when calling `embed()` /
  `rerank()` without a configured or ready model, or when the builder cannot
  load the requested backend.
- **`EmbeddingException`** (checked) – transient inference failure.
- **`RerankException`** (checked) – transient inference failure.

---

## Mode 2: Java 8 Sidecar Bridge (for Java 8 host applications)

Java 8 applications cannot use Java 21 APIs directly. Instead they communicate
with a Java 21 sidecar process over JSON-RPC (stdin/stdout):

```
Java 8 host app
  → directml-sidecar-client-java8 (pure Java 8 JSON-RPC client)
  → spawns a Java 21 sidecar process
  → directml-sidecar (JSON-RPC adapter)
  → WinDirectMlRuntime (same API as Mode 1)
  → directml-encoder / directml-windows-bindings
```

### Maven coordinates (Java 8 client)

```xml
<dependency>
    <groupId>com.aresstack</groupId>
    <artifactId>directml-sidecar-client-java8</artifactId>
    <version>${windirectml.version}</version>
</dependency>
```

The Java 8 client has **no dependency** on Java 21 runtime modules. It only
depends on `directml-config` (shared protocol types) and Jackson for JSON.

### Sidecar protocol

See [`directml-sidecar/PROTOCOL.md`](../directml-sidecar/PROTOCOL.md) for the
full JSON-RPC 2.0 specification (methods: `embed`, `embedBatch`, `rerank`,
`health`, `summarize`, `shutdown`, `cancel`).

---

## Architecture diagram

```
┌─────────────────────────────────────────────────────────┐
│  Java 21 Application (Mode 1)                           │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  WinDirectMlRuntime  (directml-runtime)           │  │
│  │    .embed()  .embedBatch()  .rerank()             │  │
│  └────────────────────────┬──────────────────────────┘  │
│                           │                             │
│  ┌────────────────────────▼──────────────────────────┐  │
│  │  directml-encoder                                 │  │
│  │    EmbeddingModel / Reranker implementations      │  │
│  └────────────────────────┬──────────────────────────┘  │
│                           │                             │
│  ┌────────────────────────▼──────────────────────────┐  │
│  │  directml-windows-bindings (FFM / DirectML)       │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Java 8 Application (Mode 2)                            │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  directml-sidecar-client-java8 (JSON-RPC client)  │  │
│  └────────────────────────┬──────────────────────────┘  │
│                           │ stdin/stdout JSON-RPC        │
│  ┌────────────────────────▼──────────────────────────┐  │
│  │  directml-sidecar (adapter)                       │  │
│  │    → delegates to WinDirectMlRuntime              │  │
│  └────────────────────────┬──────────────────────────┘  │
│                           │                             │
│  ┌────────────────────────▼──────────────────────────┐  │
│  │  directml-runtime / directml-encoder / bindings   │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Module dependency graph

```
directml-runtime
  ├── directml-encoder
  │     ├── directml-config
  │     └── directml-windows-bindings
  └── directml-config

directml-sidecar (adapter)
  ├── directml-runtime
  ├── directml-inference (Phi-3 summarizer, experimental)
  └── directml-sidecar-client-java8 (shared protocol types)

directml-sidecar-client-java8  (Java 8, no Java 21 deps)
  └── directml-config
```
