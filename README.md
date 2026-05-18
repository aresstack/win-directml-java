# win-directml-java

Pure Java DirectML runtime and sidecar for Windows. Talks JSON-RPC 2.0 over
stdin/stdout, so a Java 8 host application can drive a Java 21 DirectML
sidecar without JNI, JNA or ONNX Runtime.

This repository is a cleaned extraction of the DirectML and inference-relevant
code from `aresstack/win-acp-java`. ACP, MCP, graph routing and agent-host
layers are intentionally **not** included.

## Architecture

```text
Java-8 host application
    │ JSON-RPC 2.0, one message per line
    │ stdin / stdout (logs on stderr)
    ▼
Java-21 DirectML sidecar
    │ Java FFM (preview)
    ▼
Windows 11 DirectML / D3D12 / DXGI
```

### Non-Goals

- No ACP, no MCP
- No ONNX Runtime
- No JNA, no JNI
- No universal ONNX/GGUF/Transformers runner
- No Java CPU transformer runtime

## Modules

```text
directml-config            Minimal inference configuration
directml-windows-bindings  Java-21 FFM bindings (DXGI, D3D12, DirectML, COM/HRESULT)
directml-inference         Decoder runtime (Phi-3 today) + Summarizer use-case API
directml-encoder           Encoder runtime skeleton (MiniLM, E5, JinaBERT planned)
directml-sidecar           JSON-RPC 2.0 sidecar entry point + dispatcher + handlers
```

## Status

| Area                                        | Status |
|---------------------------------------------|--------|
| Phi-3 summarizer                            | ✅ working (CPU + DirectML hybrid) |
| JSON-RPC protocol                           | ✅ formalized (`directml-sidecar/PROTOCOL.md`) |
| `health`, `summarize`, `shutdown`, `cancel` | ✅ wired |
| `embed`                                     | ⏳ stub (`-32005 Not implemented`) until encoder runtime lands |
| MiniLM encoder                              | ⏳ architecture descriptor only |

## Requirements

- Windows 11 with DirectML support
- JDK 21 (Zulu, Temurin, …)
- Gradle wrapper from this repository

FFM is still a preview feature, so builds and runs use:

```text
--enable-preview
--enable-native-access=ALL-UNNAMED
```

## Build

PowerShell:

```powershell
$env:JAVA_HOME = "C:\Program Files\Zulu\zulu-21"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
./gradlew.bat clean build
```

See [`howtobuild.md`](howtobuild.md) for details.

## Run the sidecar

```powershell
./gradlew.bat :directml-sidecar:run
```

Talk to it via JSON-RPC 2.0 (one message per line on stdout):

```json
{"jsonrpc":"2.0","id":"h1","method":"health","params":{}}
{"jsonrpc":"2.0","id":"s1","method":"summarize","params":{"text":"...","maxTokens":256}}
{"jsonrpc":"2.0","id":"x","method":"shutdown","params":{}}
```

Protocol details: [`directml-sidecar/PROTOCOL.md`](directml-sidecar/PROTOCOL.md).

## Roadmap

1. ✅ Phi-3 summarizer sidecar.
2. ✅ Stable JSON-RPC 2.0 protocol over stdin/stdout.
3. ⏳ DirectML encoder runtime for `all-MiniLM-L6-v2` (then E5, JinaBERT).
4. ⏳ Reranker encoder support.
5. ⏳ Additional decoder LLM families after the encoder path is stable.

Issue backlog: [`win-directml-java-issues.md`](win-directml-java-issues.md).
