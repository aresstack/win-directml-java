# win-directml-java

Pure Java DirectML runtime and sidecar for Windows.

This repository is a cleaned extraction of the DirectML and inference-relevant code from `aresstack/win-acp-java`.
ACP, MCP, graph routing, and agent-host layers are intentionally not included.

## Scope

- Java 21 FFM bindings for Windows system DLLs:
  - `dxgi.dll`
  - `d3d12.dll`
  - `DirectML.dll`
- Phi-3 DirectML inference path from the prototype
- JSON-line sidecar entry point for process-based integration from a Java 8 host application
- Foundation for future DirectML encoder families for embeddings and reranking

## Modules

```text
directml-config            Minimal inference configuration
directml-windows-bindings  Java 21 FFM bindings and DirectML/D3D12 helpers
directml-inference         Inference API and current Phi-3 runtime
directml-sidecar           Stdin/stdout JSON-line sidecar entry point
```

## Requirements

- Windows 11 or compatible Windows version with DirectML
- JDK 21
- Gradle wrapper from this repository

FFM is a preview feature in JDK 21, so builds and runs use:

```text
--enable-preview
--enable-native-access=ALL-UNNAMED
```

## Run the Phi-3 sidecar

```bash
./gradlew :directml-sidecar:run
```

The sidecar reads one JSON command per line from stdin and writes one JSON response per line to stdout.
Logs must go to stderr.

Example request:

```json
{"prompt":"Fasse DirectML in zwei Sätzen zusammen.","maxTokens":128}
```

## Roadmap

1. Stabilize the Phi-3 summarizer sidecar.
2. Introduce a stable JSON-RPC 2.0 protocol over stdin/stdout.
3. Add DirectML encoder runtime families for embedding models.
4. Add reranker encoder support.
5. Add decoder LLM families after the encoder path is stable.
