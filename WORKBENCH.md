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
to the sidecar only via stdin/stdout JSON-RPC.

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
| Config & Control   | Java exe, sidecar jar, model dir, `embed.backend`, DLL override, extra JVM args, timeout. Buttons: `Start / Stop / Restart Sidecar`, `Health`, `Clear Logs`. Live command-line preview. |
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

`embed.model` dropdown policy:

- always includes legacy family aliases: `minilm`, `e5`
- includes full model IDs only when they are runtime-selectable embeddings
  (`useCase=embedding` and `embedFamily != null` in `EmbeddingModelRegistry`)
- excludes decoder/summarizer IDs and planned/unimplemented embeddings so
  the UI does not present non-runnable selections as valid

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
config.

setJavaExecutable("C:\\Program Files\\Java\\jdk-21\\bin\\java.exe");
config.

setSidecarJarPath("directml-sidecar.jar");
config.

setModelDirectory("model/all-MiniLM-L6-v2");
config.

setEmbedBackend("directml");          // or "auto" / "cpu"

SidecarClient client = new SidecarClient(config);
try{
        client.

start();

HealthResult h = client.health();
EmbeddingResult e = client.embed("hello world");
double cos = EmbeddingResult.cosine(e.getVector(),
        client.embed("hi").getVector());

    try{
SummaryResult s = client.summarize("long input…", 256);
    }catch(
JsonRpcError err){
        // summarizer not loaded → -32005 / not implemented
        }
        }finally{
        client.

shutdown();
}
```

Threading rules for Swing host applications:

- never call `health()` / `embed()` / `summarize()` from the EDT
- wrap them in `SwingWorker.doInBackground()` like the workbench does
- `shutdown()` is safe to call multiple times

## Tests

No GPU is required. Both modules are tested headlessly:

```powershell
.\gradlew.bat :directml-sidecar-client-java8:test
.\gradlew.bat :directml-sidecar-workbench:test
```

- `JsonRpcRequestTest` / `JsonRpcResponseTest` – framing round-trip.
- `SidecarProcessCommandLineTest` – verifies the spawned argv.
- `SidecarClientFakeProcessTest` – end-to-end JSON-RPC against an
  in-memory fake `SidecarProcess` (health, timeout, JSON-RPC error,
  exit code).
- `WorkbenchModelTest` – headless model logic.
- `WorkbenchFrameSmokeTest` – Swing instantiation; auto-skipped on
  truly headless build hosts.

## Java compatibility

`directml-sidecar-client-java8` and `directml-sidecar-workbench` use
`options.release.set(8)` in their `build.gradle`, so the compiler rejects
any accidental use of Java-9+ APIs. The library is therefore safe to
consume from a Java-8 host application.
