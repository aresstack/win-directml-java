# DirectML Sidecar Workbench

Java-8 Swing workbench + Java-8 client library for manually exercising the
Java-21 DirectML sidecar. Useful for end-to-end testing of `health`,
`embed`, and `summarize` without touching any DirectML, FFM, or encoder
class from the host process.

## Modules

```
directml-sidecar-client-java8     pure Java 8, JSON-RPC client lib
directml-sidecar-workbench        Java 8 Swing developer UI
```

Both modules **must not** depend on `directml-windows-bindings`,
`directml-encoder`, `directml-inference`, or `directml-sidecar`. They talk
to the sidecar only via stdin/stdout JSON-RPC. The Workbench may use shared
Java-8 helper classes from the client module, such as the CPU-only model
validator described in [`docs/model-validation.md`](docs/model-validation.md).

## Build & run

Build the sidecar first (still Java 21):

```powershell
.\gradlew.bat :directml-sidecar:jar
```

Build the Java-8 modules and run the workbench:

```powershell
.\gradlew.bat :directml-sidecar-workbench:run
```

The window opens with seven tabs:

| Tab                | Purpose                                                                                                                                                                                 |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Config & Control   | Java exe, sidecar jar, model dir, `embed.backend`, DLL override, extra JVM args, timeout. Buttons: `Start / Stop / Restart Sidecar`, `Health`, `Validate Models`, `Clear Logs`. Live command-line preview. `Validate Models` runs the CPU-only model directory check without starting the sidecar. |
| Health             | `sidecarRunning`, `embeddingReady`, `embeddingBackend`, `modelLoaded`, `mode`, `lastError` + raw JSON.                                                                                  |
| Embeddings         | Two text fields (A/B), `Embed A`, `Embed B`, `Cosine Similarity`, dimension/timing/backend readout.                                                                                     |
| Summarize          | Input text, `maxTokens` spinner, summary output, raw response. Shows JSON-RPC errors clearly when no summarizer is loaded.                                                              |
| JSON-RPC Inspector | Last raw request and last raw response, refreshed every 500 ms.                                                                                                                         |
| stderr Log         | Live tail of the sidecar's stderr (separate from stdout).                                                                                                                               |
| Integration Help   | Java-8 sample code that uses the same client library.                                                                                                                                   |

## Architecture

```
Swing Workbench (Java 8)
  → SidecarClient (Java 8)
  → SidecarProcess + stdin/stdout JSON-RPC
  → Java-21 DirectML sidecar
  → DirectML / Phi-3 / MiniLM
```

The workbench **never** loads any DirectML or model class. Everything goes
through the JSON-RPC stream owned by `SidecarClient`.

`SidecarProcess`:

- spawns `java --enable-preview --enable-native-access=ALL-UNNAMED -Dembed.backend=… -jar directml-sidecar.jar`
- keeps stdout strictly line-oriented JSON-RPC
- buffers stderr separately for the UI
- supports graceful stop (`destroy`) and force-stop (`destroyForcibly`)

`SidecarClient`:

- monotonic request id → `CompletableFuture<JsonRpcResponse>` registry
- background stdout reader thread classifies lines as response / notification
- `health()`, `embed()`, `summarize()`, `shutdown()` are synchronous; the
  workbench wraps them in `SwingWorker` so the EDT never blocks
- exposes `getLastRawRequest()`, `getLastRawResponse()`, `getStderrSnapshot()`
  for the JSON-RPC Inspector and stderr Log

## Java-8 host integration

The exact same client library can be embedded into any Java-8 application:

```java
SidecarClientConfig config = new SidecarClientConfig();
config.setJavaExecutable("C:\\Program Files\\Java\\jdk-21\\bin\\java.exe");
config.setSidecarJarPath("directml-sidecar-all.jar");
config.setModelDirectory("model/all-MiniLM-L6-v2");
config.setEmbedBackend("directml");

try (SidecarClient client = new SidecarClient(config)) {
    client.start();
    HealthResult h = client.health();
    EmbeddingResult v = client.embed("hello world");
}
```

See also [`examples/java8-client`](examples/java8-client) for a runnable sample.
