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
- No *production* Java CPU transformer runtime. The CPU MiniLM path
  (`CpuMiniLmEncoder`) exists **only as a correctness reference** for the
  DirectML migration and is not intended to scale.

## Modules

```text
directml-config            Minimal inference configuration
directml-windows-bindings  Java-21 FFM bindings (DXGI, D3D12, DirectML, COM/HRESULT)
directml-inference         Decoder runtime (Phi-3 today) + Summarizer use-case API
directml-encoder           Encoder runtime skeleton (MiniLM, E5, JinaBERT planned)
directml-sidecar           JSON-RPC 2.0 sidecar entry point + dispatcher + handlers
```

## Status

| Area                                                             | Status                                                                         |
|------------------------------------------------------------------|--------------------------------------------------------------------------------|
| Phi-3 summarizer                                                 | ✅ working (CPU + DirectML hybrid)                                              |
| JSON-RPC protocol                                                | ✅ formalized (`directml-sidecar/PROTOCOL.md`)                                  |
| `health`, `summarize`, `shutdown`, `cancel`                      | ✅ wired                                                                        |
| `embed`                                                          | ✅ wired – CPU reference `MiniLM` encoder returns real 384-dim vectors          |
| Runtime-Core API (D3D12/DML context, Tensor, GpuBuffer, Kernels) | 🟡 `DirectMlContextImpl` + `DefaultGpuBuffer` live (roundtrip-tested on real GPU); DirectML kernel impls pending |
| SafetensorsReader                                                | ✅ implemented + tested (F32/F16/BF16/I64/I32/I8/U8, lenient on unknown dtypes) |
| WordPieceTokenizer                                               | ✅ implemented + tested (BERT-uncased family)                                   |
| Mean Pooling + L2                                                | ✅ CPU reference impl + tests                                                   |
| MiniLM encoder runtime                                           | ✅ CPU forward pass (`CpuMiniLmEncoder`) – DirectML kernel migration pending    |
| Sidecar lifecycle tests                                          | ✅ end-to-end via piped streams                                                 |
| Phi-3 benchmark harness                                          | ✅ runnable (`Phi3Benchmark`)                                                   |
| E5 / Reranker / further decoders                                 | 📄 concept docs in `docs/`                                                     |

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

## Embeddings (`embed`)

`embed` is backed by a pure-Java CPU implementation of
`sentence-transformers/all-MiniLM-L6-v2` (`CpuMiniLmEncoder`). Behaviour is
deliberately *correct first, fast later*: a single sentence at sequence length
≤ 128 takes < 100 ms on a modern desktop. The CPU pass serves as the reference
for the upcoming DirectML kernel migration.

Fetch the model once (≈ 90 MB):

```powershell
pwsh scripts/download-minilm.ps1
```

Then `embed("…")` returns a real 384-dim, L2-normalised vector. Reference tests
in `EmbeddingReferenceTest` validate cosine separation between semantically
related vs. unrelated sentences.

## Roadmap

1. ✅ Phi-3 summarizer sidecar.
2. ✅ Stable JSON-RPC 2.0 protocol over stdin/stdout.
3. ✅ CPU reference encoder for `all-MiniLM-L6-v2` (validated via cosine reference tests).
4. ✅ Concrete `DirectMlContextImpl` + `DefaultGpuBuffer` with explicit D3D12 state machine
   (`COMMON → COPY_DEST → UAV → COPY_SOURCE → UAV`), validated by GPU upload/download roundtrip test.
5. ⏳ `DirectMlLinearKernel` (DML_GEMM_OPERATOR_DESC) – first real DirectML operator.
6. ⏳ Migrate MiniLM forward pass to DirectML kernels (`LayerNormKernel`, `GeluKernel`,
   `AttentionKernel`); reference test `CpuMiniLmEncoder.embed(t)` vs. `DirectMlMiniLmEncoder.embed(t)`
   cosine > 0.99.
7. ⏳ E5 and JinaBERT encoders on the same runtime core.
8. ⏳ Reranker encoder support.
9. ⏳ Additional decoder LLM families after the encoder path is stable.

Issue backlog: [`win-directml-java-issues.md`](win-directml-java-issues.md).
