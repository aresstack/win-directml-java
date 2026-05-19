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

| Area                                                    | Status                                                                                                                                                                                                                                                                                                                                                               |
|---------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Phi-3 summarizer                                        | ✅ working (CPU + DirectML hybrid)                                                                                                                                                                                                                                                                                                                                    |
| JSON-RPC protocol                                       | ✅ formalized (`directml-sidecar/PROTOCOL.md`)                                                                                                                                                                                                                                                                                                                        |
| `health`, `summarize`, `shutdown`, `cancel`             | ✅ wired                                                                                                                                                                                                                                                                                                                                                              |
| `embed`                                                 | ✅ wired through the backend switch – `CpuMiniLmEncoder` and `DirectMlMiniLmEncoder` selectable via `-Dembed.backend=cpu\|directml\|auto`; both return real 384-dim L2-normalised vectors                                                                                                                                                                              |
| Runtime-Core API (D3D12/DML context, Tensor, GpuBuffer) | ✅ `DirectMlContextImpl` + `DefaultGpuBuffer` (GPU roundtrip-tested)                                                                                                                                                                                                                                                                                                  |
| DirectML kernels (`Linear`, `LayerNorm`, `GELU`)        | ✅ `DirectMlLinearKernel`, `DirectMlLayerNormKernel`, `GeluKernel.create(...)` – picks `DirectMlGeluKernel` (native `DML_OPERATOR_ACTIVATION_GELU`, FL ≥ 5.1) on modern DLLs and `DirectMlCompositeGeluKernel` (`ERF + IDENTITY + MULTIPLY`, FL 2.0) on the in-box Windows-11 RTM `DirectML.dll` 1.8.0. Both paths CPU-reference-tested on real GPU (cos = 1.000000). |
| DirectML kernels (`Softmax`)                            | ✅ `DirectMlSoftmaxKernel` (`DML_OPERATOR_ACTIVATION_SOFTMAX`, FL 2.0) – row-wise, CPU-reference-tested, runs on every shipped `DirectML.dll`                                                                                                                                                                                                                         |
| DirectML kernels (`Attention`)                          | ✅ `DirectMlAttentionKernel` – composite SDPA over `GEMM` ×2 + optional masked `ELEMENT_WISE_ADD` + `Softmax`, all FL 2.0 (runs on every shipped `DirectML.dll`, in-box 1.8.0 included). CPU-reference-tested on real GPU, with and without padding mask.                                                                                                             |
| DirectML kernels (head layout `[S,H,D]↔[H,S,D]`)        | ✅ `DirectMlHeadLayoutKernel` via `DML_OPERATOR_ELEMENT_WISE_IDENTITY` + strided input view; both directions and round-trip bit-exact. See [`docs/head-layout-convention.md`](docs/head-layout-convention.md).                                                                                                                                                        |
| SafetensorsReader                                       | ✅ implemented + tested (F32/F16/BF16/I64/I32/I8/U8, lenient on unknown dtypes)                                                                                                                                                                                                                                                                                       |
| WordPieceTokenizer                                      | ✅ implemented + tested (BERT-uncased family)                                                                                                                                                                                                                                                                                                                         |
| Mean Pooling + L2                                       | ✅ CPU reference impl + tests                                                                                                                                                                                                                                                                                                                                         |
| MiniLM encoder runtime                                  | ✅ `DirectMlMiniLmEncoder` implemented: 6-layer DirectML transformer stack active end-to-end (Q/K/V/Linear + composite/native GELU + LayerNorm + attention + MLP + residuals). Selectable via sidecar backend switch (`-Dembed.backend=cpu\|directml\|auto`). Pad-bucket cache (S ∈ {64,128,256,512}) coalesces tokenizer lengths onto ≤ 4 cached stacks. Reference test confirms `cos(CPU, DirectML) = 1.000000` on Windows 11 in-box FL 5.0. MeanPool runs on DirectML (`DirectMlMeanPoolKernel`, GEMM FL-1.0); L2 remains CPU-side on the final 384-float vector.                |
| Sidecar lifecycle tests                                 | ✅ end-to-end via piped streams                                                                                                                                                                                                                                                                                                                                       |
| Phi-3 benchmark harness                                 | ✅ runnable (`Phi3Benchmark`)                                                                                                                                                                                                                                                                                                                                         |
| E5 / Reranker / further decoders                        | 📄 concept docs in `docs/`                                                                                                                                                                                                                                                                                                                                           |

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

### Optional: opt-in NuGet redist downloader

If your application wants to enable fused FL ≥ 5.1 kernels even on a host
whose in-box `DirectML.dll` is the 1.8.0 RTM baseline, it may download the
official `Microsoft.AI.DirectML` NuGet package into its own per-user
settings folder. The `directml-windows-bindings` module ships a generic
NuGet installer (`com.aresstack.windirectml.runtime.redist.*`) for exactly
this purpose:

```java
import com.aresstack.windirectml.runtime.redist.AppSettingsPaths;
import com.aresstack.windirectml.runtime.redist.DirectMlRedistInstaller;

// ONLY call this after the user has explicitly opted in, e.g. through a
// settings checkbox or first-run dialog. Never download silently.
Path appSettings = AppSettingsPaths.forApp("my-app");
DirectMlRedistInstaller installer = new DirectMlRedistInstaller(appSettings);
Path dll = installer.installAndUse("1.15.4", "x64");   // FL 6.4 on every Win11
// ...subsequent DirectMlBindings.createDevice(...) loads the redist DLL.
```

`installAndUse` is equivalent to calling `install(...)` and then setting
`-D` system property `windirectml.directml.dll` to the returned path. It
must be called **before** the first use of `DirectMlBindings`, otherwise
the symbol lookup is already cached against the in-box DLL.

Layout under `%LOCALAPPDATA%\<app-name>`:

```text
nuget-cache/Microsoft.AI.DirectML/<version>/Microsoft.AI.DirectML.<version>.nupkg
native/directml/<version>/<arch>-win/DirectML.dll
```

Supported architectures: `x64`, `x86`, `arm64`, `arm`. The installer is
idempotent (cached `.nupkg` is reused; extract is a no-op if the file is
already in place with the expected size) and refuses to write outside the
app settings folder (ZIP-slip protection).

A real end-to-end test against `nuget.org` is included but opt-in:

```powershell
./gradlew.bat :directml-windows-bindings:test `
    --tests *DirectMlRedistDownloadIT `
    "-Dwindirectml.redist.itest=true"
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

## Java-8 Workbench

For interactive testing of the sidecar from a Java-8 host environment
(stdin/stdout JSON-RPC, no FFM, no DirectML in the host), see the
companion modules and Swing workbench:

```powershell
./gradlew.bat :directml-sidecar:jar
./gradlew.bat :directml-sidecar-workbench:run
```

Documentation: [`WORKBENCH.md`](WORKBENCH.md).

## Embeddings (`embed`)

`embed` is backed by `sentence-transformers/all-MiniLM-L6-v2`. Two
interchangeable encoder implementations exist behind the same
`EmbeddingModel` API:

* **`DirectMlMiniLmEncoder`** – the intended product path. Hybrid
  CPU/GPU pipeline:
  * **CPU**: tokenization + word/position/tokenType embedding lookup
    (small int→float gather, no benefit from GPU dispatch).
  * **DirectML**: embedding LayerNorm + the full 6-layer BERT encoder
    (Q/K/V/Linear + composite/native GELU + LayerNorm, head-layout
    reshuffles, attention, MLP, residuals).
  * **CPU**: mean-pool over `attentionMask` + L2 normalize (final
    384-float vector) — scheduled to move to DirectML next.

  Reference test `DirectMlMiniLmEmbeddingReferenceTest` confirms
  `cos(CPU, DirectML) = 1.000000` over the full corpus. Works on every
  shipped `DirectML.dll`, including Windows 11 RTM in-box 1.8.0 (FL 5.0),
  via the composite GELU fallback in `GeluKernel.create(...)`.
* **`CpuMiniLmEncoder`** – pure-Java CPU reference / debug / fallback.
  Correct-first, fast-later: a single sentence at sequence length ≤ 128
  takes < 100 ms on a modern desktop.

Fetch the model once (≈ 90 MB):

```powershell
pwsh scripts/download-minilm.ps1
```

`embed("…")` then returns a real 384-dim, L2-normalised vector regardless
of the active backend.

### Backend selection

The sidecar picks between the two encoders via `-Dembed.backend`:

```powershell
./gradlew.bat :directml-sidecar:run -Dembed.backend=auto       # default: try DirectML, fall back to CPU with warn log
./gradlew.bat :directml-sidecar:run -Dembed.backend=directml   # force DirectMlMiniLmEncoder; exit code 3 if unavailable
./gradlew.bat :directml-sidecar:run -Dembed.backend=cpu        # force CpuMiniLmEncoder (reference / debug path)
```

A forced mode (`cpu` or `directml`) exits with code `3` if its encoder
cannot be loaded – including the case where no MiniLM model directory is
present at all. `auto` falls back silently to CPU and records the
DirectML error message in `lastError`. The active backend is reported in
the `health` response as `embeddingBackend` (`cpu`, `directml`, `none`,
or `error`) together with `embeddingReady`.

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
7. ✅ `GeluKernel.create(...)` – auto-selects the GPU GELU strategy:
    * `DirectMlGeluKernel` – native fused `DML_OPERATOR_ACTIVATION_GELU` (op 157, requires `DML_FEATURE_LEVEL_5_1`).
    * `DirectMlCompositeGeluKernel` – portable fallback built from `ELEMENT_WISE_ERF(scale=1/√2)` +
      `ELEMENT_WISE_IDENTITY(scale=0.5, bias=0.5)` + `ELEMENT_WISE_MULTIPLY`, all FL 2.0. Runs on every shipped
      `DirectML.dll`, including the Windows 11 RTM in-box 1.8.0 (FL 5.0).
      Both paths are CPU-reference-tested on real GPU (cos(CPU, DirectML) = 1.000000 across the full MiniLM encoder
      reference corpus).
8. ✅ `DirectMlSoftmaxKernel` – `DML_OPERATOR_ACTIVATION_SOFTMAX` (op 48, FL 2.0). Row-wise normalisation over the
   innermost axis; CPU-reference-tested. Building block for the attention kernel and reranker logit heads.
9. ✅ `DirectMlAttentionKernel` – scaled-dot-product multi-head attention composed of `GEMM` (Q·Kᵀ, scaled via
   `Alpha`), optional masked `ELEMENT_WISE_ADD` (mask broadcast via strides `[S,0,0,1]`), `Softmax`, second `GEMM`
   (·V). All four sub-ops are FL 2.0, so the kernel runs on every shipped `DirectML.dll`, including the
   Windows 11 RTM in-box 1.8.0. CPU-reference-tested on real GPU with and without padding mask
   (tolerance 5e-4). Native fused `DML_OPERATOR_MULTIHEAD_ATTENTION` (op 164, FL 6.1) reserved as an optional
   fast path.
10. ✅ `DirectMlMiniLmEncoder` – kernels wired into the full encoder pipeline.
    Reference test `DirectMlMiniLmEmbeddingReferenceTest` confirms
    `cos(CpuMiniLmEncoder.embed(t), DirectMlMiniLmEncoder.embed(t)) = 1.000000`
    over the full corpus.
    **Step 10a:** ✅ `DirectMlMiniLmLayerBlock` – one full encoder layer (Q/K/V Linear → head layout → Attention → head
    layout → Wo Linear → residual+LN → MLP w/ GELU → residual+LN). Synthetic CPU-vs-DirectML compare at tolerance 2e-3 (
    `DirectMlMiniLmLayerBlockTest`). Runs on FL ≥ 5.0 via the GELU strategy switch in step 7.
    **Step 10b:** ✅ `DirectMlMiniLmEncoderStack` – multi-layer wiring (6 layers + token-type/position/word embeddings +
    embedding LN), reference-tested in `DirectMlMiniLmEncoderStackTest`.
    **Step 10c:** ✅ `DirectMlMiniLmEncoder.embed(t)` – mean-pool + L2 on CPU after the DirectML stack, returning a
    384-dim L2-normalised vector.
11. ✅ Sidecar embedding switch (`-Dembed.backend=cpu|directml|auto`) – wires `DirectMlMiniLmEncoder` and
    `CpuMiniLmEncoder` behind one JSON-RPC endpoint; forced modes fail visibly with exit code 3.
12. ✅ Pad-bucket cache for the DirectML encoder (`S ∈ {64, 128, 256, 512}`) – arbitrary tokenizer lengths are padded
    up to the smallest matching bucket; the encoder caches at most four form-bound stacks instead of one per actual
    sequence length. Padded positions are masked out in attention (`-1e9`) and ignored by MeanPool via the original
    `attentionMask`, so `cos(CPU, DirectML)` is unchanged. Validated by `DirectMlMiniLmEncoderBucketTest` and the
    existing reference test (`cachedStackCount() == 1` for the short corpus).
13. ✅ Mean-pool on DirectML (`DirectMlMeanPoolKernel`, `DML_OPERATOR_GEMM` FL-1.0) – per-token weights
    `w[t] = m[t]/Σm` are pre-normalised on the CPU, the GEMM `y[1,H] = w[1,S]·x[S,H]` then collapses the pooling
    into a single FL-1.0 dispatch. The PCIe read-back per inference shrinks from `B·H` to `H` floats (≈ 256× less
    on bucket `S=256, H=384`). L2-Normalize stays on the CPU because the operation on a single 384-float vector
    is microseconds and would otherwise require Reduce/Divide ops that are not guaranteed on FL 5.0 in-box DLLs.
    Validated by `DirectMlMeanPoolKernelTest` and the existing parity test against the CPU encoder
    (`DirectMlMiniLmEmbeddingReferenceTest`).
14. ⏳ E5 and JinaBERT encoders on the same runtime core.
15. ⏳ Reranker encoder support.
16. ⏳ Additional decoder LLM families after the encoder path is stable.

Issue backlog: [`win-directml-java-issues.md`](win-directml-java-issues.md).
