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

| Area                                                    | Status                                                                                                                                                     |
|---------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Phi-3 summarizer                                        | ✅ working (CPU + DirectML hybrid)                                                                                                                          |
| JSON-RPC protocol                                       | ✅ formalized (`directml-sidecar/PROTOCOL.md`)                                                                                                              |
| `health`, `summarize`, `shutdown`, `cancel`             | ✅ wired                                                                                                                                                    |
| `embed`                                                 | ✅ wired – CPU reference `MiniLM` encoder returns real 384-dim vectors                                                                                      |
| Runtime-Core API (D3D12/DML context, Tensor, GpuBuffer) | ✅ `DirectMlContextImpl` + `DefaultGpuBuffer` (GPU roundtrip-tested)                                                                                        |
| DirectML kernels (`Linear`, `LayerNorm`, `GELU`)        | ✅ `DirectMlLinearKernel`, `DirectMlLayerNormKernel`, `DirectMlGeluKernel` (native `DML_OPERATOR_ACTIVATION_GELU`, requires `DML_FEATURE_LEVEL_5_1`) – all CPU-reference-tested on real GPU |
| DirectML kernels (`Softmax`)                            | ✅ `DirectMlSoftmaxKernel` (`DML_OPERATOR_ACTIVATION_SOFTMAX`, FL 2.0) – row-wise, CPU-reference-tested, runs on every shipped `DirectML.dll`                |
| DirectML kernels (`Attention`)                          | ⏳ next sprint                                                                                                                                              |
| SafetensorsReader                                       | ✅ implemented + tested (F32/F16/BF16/I64/I32/I8/U8, lenient on unknown dtypes)                                                                             |
| WordPieceTokenizer                                      | ✅ implemented + tested (BERT-uncased family)                                                                                                               |
| Mean Pooling + L2                                       | ✅ CPU reference impl + tests                                                                                                                               |
| MiniLM encoder runtime                                  | ✅ CPU forward pass (`CpuMiniLmEncoder`) – DirectML kernel migration pending                                                                                |
| Sidecar lifecycle tests                                 | ✅ end-to-end via piped streams                                                                                                                             |
| Phi-3 benchmark harness                                 | ✅ runnable (`Phi3Benchmark`)                                                                                                                               |
| E5 / Reranker / further decoders                        | 📄 concept docs in `docs/`                                                                                                                                 |

## Requirements

- Windows 11 with DirectML support
- JDK 21 (Zulu, Temurin, …)
- Gradle wrapper from this repository

FFM is still a preview feature, so builds and runs use:

```text
--enable-preview
--enable-native-access=ALL-UNNAMED
```

## Runtime configuration

| System property                 | Default                       | Effect                                                                                                                                                                                                                                             |
|---------------------------------|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `windirectml.debug`             | `false`                       | Enables the D3D12 + DirectML debug layer. Drains the InfoQueue and surfaces the messages in exceptions.                                                                                                                                            |
| `windirectml.directml.dll`      | *(unset → in-box `System32`)* | Absolute path to a `DirectML.dll` to load instead of the in-box copy. Use this to ship a [Microsoft.AI.DirectML](https://www.nuget.org/packages/Microsoft.AI.DirectML/) redistributable next to your application (e.g. `app/native/DirectML.dll`). |
| `windirectml.dxgi.adapterIndex` | `0`                           | DXGI adapter index to bind.                                                                                                                                                                                                                        |

### Feature-level fallback

DirectML operators are gated by `DML_FEATURE_LEVEL`. At startup the runtime
logs the active level and which fused operators it enables:

```text
DirectML.dll source: DirectML.dll (system), max DML_FEATURE_LEVEL = 5.0 (raw 0x5000)
Fused-op availability: GELU(FL>=5.1)=false, MultiHeadAttention(FL>=6.1)=false
```

Reference points:

| `DirectML.dll`      | Source                             | Max FL |
|---------------------|------------------------------------|--------|
| 1.8.0 (Jan 2022)    | Windows 11 21H2/22H2 in-box        | 5.0    |
| 1.15.4.0 (Nov 2025) | Microsoft.AI.DirectML redist / WSL | 6.4    |
| 1.15.5.0 (Mar 2026) | Windows 11 24H2+ in-box            | 6.4    |

If you target the 1.8.0 baseline, fused kernels that require ≥5.1 (e.g.
GELU) are unavailable and the corresponding tests are skipped. For full
coverage on older boxes bundle a redistributable and point the property at
it:

```powershell
java -Dwindirectml.directml.dll="C:\app\native\DirectML.dll" ...
```

**Policy:** `windirectml.directml.dll` is only intended for redistributables
that **your** application ships deliberately. Do **not** point it at DLLs
installed by other software (e.g. anything under `C:\Program Files\…`); the
default behaviour of loading the in-box `System32\DirectML.dll` is the only
supported production path.

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
5. ✅ `DirectMlLinearKernel` (`DML_OPERATOR_GEMM`) – first real DirectML compute kernel, CPU-reference-tested on real
   GPU.
6. ✅ `DirectMlLayerNormKernel` (`DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION`, MVN0) – BERT-style LayerNorm,
   CPU-reference-tested.
7. ✅ `DirectMlGeluKernel` – native `DML_OPERATOR_ACTIVATION_GELU` (op 157, requires `DML_FEATURE_LEVEL_5_1`).
   CPU-reference-tested on real GPU. On the Windows 11 RTM in-box `DirectML.dll` 1.8.0 (FL 5.0) the test is skipped;
   ship a Microsoft.AI.DirectML redistributable via `-Dwindirectml.directml.dll=...` to enable the kernel there.
   A composite ERF+IDENTITY+MULTIPLY fallback on FL 2.0 primitives is tracked as a follow-up.
8. ✅ `DirectMlSoftmaxKernel` – `DML_OPERATOR_ACTIVATION_SOFTMAX` (op 48, FL 2.0). Row-wise normalisation over the
   innermost axis; CPU-reference-tested. Building block for the attention kernel and reranker logit heads.
9. ⏳ `DirectMlAttentionKernel` – scaled-dot-product multi-head attention composed of `GEMM` (Q·Kᵀ, scaled via
   `Alpha`), masked `ELEMENT_WISE_ADD`, `Softmax`, second `GEMM` (·V) and an output projection. Native fused
   `DML_OPERATOR_MULTIHEAD_ATTENTION` (op 164, FL 6.1) reserved as an optional fast path.
10. ⏳ `DirectMlMiniLmEncoder` – wire the kernels together; reference test `CpuMiniLmEncoder.embed(t)` vs.
    `DirectMlMiniLmEncoder.embed(t)` cosine > 0.99.
11. ⏳ E5 and JinaBERT encoders on the same runtime core.
12. ⏳ Reranker encoder support.
12. ⏳ Additional decoder LLM families after the encoder path is stable.

Issue backlog: [`win-directml-java-issues.md`](win-directml-java-issues.md).
