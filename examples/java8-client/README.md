# Java 8 sidecar client example

This example shows how a Java 8 host application can use the published
`directml-sidecar-client-java8` artifact to drive the Java 21 sidecar over
stdin/stdout JSON-RPC.

The host imports only:

```java
com.aresstack.windirectml.sidecar.client.*
```

It does not import `directml-encoder`, `directml-windows-bindings`, Java FFM,
DirectML, D3D12 or model-loader classes. Those stay inside the sidecar process.

## Maven Central coordinate

```gradle
dependencies {
    implementation 'com.aresstack:directml-sidecar-client-java8:0.1.0-beta.1'
}
```

```xml
<dependency>
  <groupId>com.aresstack</groupId>
  <artifactId>directml-sidecar-client-java8</artifactId>
  <version>0.1.0-beta.1</version>
</dependency>
```

For the in-repository example build this module depends on
`project(':directml-sidecar-client-java8')` so CI can compile it together with
the current source tree.

## What the example covers

- sidecar process start from Java code
- `health`
- `embed`
- `embedBatch`
- optional `rerank` when a reranker model is loaded
- timeout configuration
- `JsonRpcError`, `SidecarTimeoutException` and generic `SidecarException`
- ordered shutdown via `client.shutdown()`
- backend selection: `auto`, `directml`, `cpu`

## Build the example

```powershell
./gradlew.bat :examples:java8-client:compileJava
./gradlew.bat :examples:java8-client:installDist
```

The source is compiled with `--release 8`. Running the program without
arguments prints usage text and does not start a sidecar, which makes it safe
as a Java 8 compilation smoke test on non-Windows CI.

## Run it

First create/download a sidecar runtime jar and a model directory. For local
repository development:

```powershell
./gradlew.bat :directml-sidecar:jar
pwsh scripts/download-minilm.ps1
```

Then run the installed example:

```powershell
examples/java8-client/build/install/java8-client/bin/java8-client.bat `
    directml-sidecar/build/libs/directml-sidecar-0.1.0-SNAPSHOT.jar `
    model/all-MiniLM-L6-v2 `
    auto
```

The third argument selects the backend:

| Value      | Meaning                                                                               |
|------------|---------------------------------------------------------------------------------------|
| `auto`     | Try DirectML first, then fall back to CPU and report the reason in health.            |
| `directml` | Force DirectML; startup fails visibly if DirectML or the model cannot load.           |
| `cpu`      | Force the pure Java CPU path. Useful for CI, VMs and machines without a DirectML GPU. |

A fourth optional argument points to a reranker model directory:

```powershell
pwsh scripts/download-reranker.ps1
examples/java8-client/build/install/java8-client/bin/java8-client.bat `
    directml-sidecar/build/libs/directml-sidecar-0.1.0-SNAPSHOT.jar `
    model/all-MiniLM-L6-v2 `
    auto `
    model/cross-encoder-ms-marco-MiniLM-L-6-v2
```

## Copy-paste client skeleton

```java
SidecarClientConfig config = new SidecarClientConfig();
config.setJavaExecutable("java");
config.setSidecarJarPath("directml-sidecar.jar");
config.setModelDirectory("model/all-MiniLM-L6-v2");
config.setEmbedModel("minilm");
config.setEmbedBackend("auto");
config.setRerankBackend("auto");
config.setRequestTimeoutMillis(60000L);

SidecarClient client = new SidecarClient(config);
try {
    client.start();
    HealthResult health = client.health();
    EmbeddingResult vector = client.embed("local search query");
    BatchEmbeddingResult batch = client.embedBatch(java.util.Arrays.asList(
            "first document", "second document"));
} catch (JsonRpcError e) {
    System.err.println("JSON-RPC error " + e.getCode() + ": " + e.getMessage());
} catch (SidecarTimeoutException e) {
    System.err.println("Timeout: " + e.getMessage());
} catch (SidecarException e) {
    System.err.println("Sidecar failure: " + e.getMessage());
} finally {
    client.shutdown();
}
```

## Notes for downstream applications

- The Java 8 application owns chunking, vector storage and vector search.
- The sidecar computes embeddings and optionally reranks a top-K candidate set.
- CPU-only mode is supported and should not be presented as an error.
- DirectML is the Windows acceleration path.
- Reranker scores are raw model logits; compare them only within the same query.
