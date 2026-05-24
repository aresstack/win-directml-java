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

## Modules

```text
directml-config            Minimal inference configuration
directml-windows-bindings  Java-21 FFM bindings (DXGI, D3D12, DirectML, COM/HRESULT)
directml-inference         Decoder runtime (Phi-3 today) + Summarizer use-case API
directml-encoder           Encoder runtime skeleton (MiniLM, E5, JinaBERT planned)
directml-sidecar           JSON-RPC 2.0 sidecar entry point + dispatcher + handlers
```

## Status

| Area                                                    | Status                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|---------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Phi-3 summarizer                                        | 🧪 experimental – Phi-3-only, sidecar-only; not part of the Maven Central core release. The summarizer continues to work over the JSON-RPC `summarize` method when a Phi-3 model directory is present, but the supported release use cases are embeddings + reranking.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| JSON-RPC protocol                                       | ✅ formalized (`directml-sidecar/PROTOCOL.md`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `health`, `summarize`, `shutdown`, `cancel`             | ✅ wired                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `embed`                                                 | ✅ wired through the family + backend switch – MiniLM (`CpuMiniLmEncoder` / `DirectMlMiniLmEncoder`) and E5 (`CpuBertEncoder` / `DirectMlBertEncoder`) selectable via `-Dembed.model=minilm\|e5` (default `minilm`) and `-Dembed.backend=cpu\|directml\|auto`; `-De5.model=small-v2\|base-v2\|large-v2\|base-sts-en-de` picks the E5 variant; both families return real L2-normalised vectors                                                                                                                                                                                                                                                                                     |
| Runtime-Core API (D3D12/DML context, Tensor, GpuBuffer) | ✅ `DirectMlContextImpl` + `DefaultGpuBuffer` (GPU roundtrip-tested)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| DirectML kernels (`Linear`, `LayerNorm`, `GELU`)        | ✅ `DirectMlLinearKernel`, `DirectMlLayerNormKernel`, `GeluKernel.create(...)` – picks `DirectMlGeluKernel` (native `DML_OPERATOR_ACTIVATION_GELU`, FL ≥ 5.1) on modern DLLs and `DirectMlCompositeGeluKernel` (`ERF + IDENTITY + MULTIPLY`, FL 2.0) on the in-box Windows-11 RTM `DirectML.dll` 1.8.0. Both paths CPU-reference-tested on real GPU (cos = 1.000000).                                                                                                                                                                                                                                                                                                             |
| DirectML kernels (`Softmax`)                            | ✅ `DirectMlSoftmaxKernel` (`DML_OPERATOR_ACTIVATION_SOFTMAX`, FL 2.0) – row-wise, CPU-reference-tested, runs on every shipped `DirectML.dll`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| DirectML kernels (`Attention`)                          | ✅ `DirectMlAttentionKernel` – composite SDPA over `GEMM` ×2 + optional masked `ELEMENT_WISE_ADD` + `Softmax`, all FL 2.0 (runs on every shipped `DirectML.dll`, in-box 1.8.0 included). CPU-reference-tested on real GPU, with and without padding mask.                                                                                                                                                                                                                                                                                                                                                                                                                         |
| DirectML kernels (head layout `[S,H,D]↔[H,S,D]`)        | ✅ `DirectMlHeadLayoutKernel` via `DML_OPERATOR_ELEMENT_WISE_IDENTITY` + strided input view; both directions and round-trip bit-exact. See [`docs/head-layout-convention.md`](docs/head-layout-convention.md).                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| SafetensorsReader                                       | ✅ implemented + tested (F32/F16/BF16/I64/I32/I8/U8, lenient on unknown dtypes)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| WordPieceTokenizer                                      | ✅ implemented + tested (BERT-uncased family)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Mean Pooling + L2                                       | ✅ CPU reference + DirectML kernels: `DirectMlMeanPoolKernel`, `DirectMlL2NormalizeKernel`, `DirectMlBatchedMeanPoolKernel` and `DirectMlBatchedL2NormalizeKernel` implemented and tested (FL-1.0, GEMM + SQRT + DIVIDE composite). Single-vector and batched `embedBatch` paths both run pool + normalize on the GPU; PCIe read-back is `[H]` (single) or `[N,H]` (batch).                                                                                                                                                                                                                                                                                                       |
| MiniLM encoder runtime                                  | ✅ `DirectMlMiniLmEncoder` implemented: 6-layer DirectML transformer stack active end-to-end (Q/K/V/Linear + composite/native GELU + LayerNorm + attention + MLP + residuals). Selectable via sidecar backend switch (`-Dembed.backend=cpu\|directml\|auto`). Pad-bucket cache (S ∈ {64,128,256,512}) coalesces tokenizer lengths onto ≤ 4 cached stacks. Reference test confirms `cos(CPU, DirectML) = 1.000000` on Windows 11 in-box FL 5.0. MeanPool (`DirectMlMeanPoolKernel`, GEMM FL-1.0) and L2-Normalize (`DirectMlL2NormalizeKernel`, GEMM + SQRT + DIVIDE composite, FL-1.0) both run on DirectML; the final 384-float vector is the only PCIe read-back per inference. |
| Sidecar lifecycle tests                                 | ✅ end-to-end via piped streams                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Phi-3 benchmark harness                                 | ✅ runnable (`Phi3Benchmark`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| E5 encoder runtime                                      | ✅ `E5Encoders.loadCpu` / `loadDirectMl` – CPU and DirectML paths share the same generic BERT stack as MiniLM and propagate the `query:` / `passage:` prefix policy; selectable via `-Dembed.model=e5` and `-De5.model=<variant>`.                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Reranker (cross-encoder) runtime                        | ✅ `BertCrossEncoderRerankers.loadCpu` / `loadDirectMl` – pair-tokenised cross-encoder with classifier head, exposed as `rerank` over JSON-RPC. CPU + DirectML backends, selectable via `-Drerank.backend=cpu\|directml\|auto` with the same forced-mode semantics as `embed.backend`. `scripts/download-reranker.ps1` drives the default `cross-encoder/ms-marco-MiniLM-L-6-v2` and `RerankerRealModelReferenceTest` validates CPU↔DirectML score parity on that real checkpoint.                                                                                                                                                                                                |
| Further decoders                                        | 📄 concept docs in `docs/`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |

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

> **Note on `summarize`.** Summarization support is **experimental** and
> currently limited to the existing Phi-3-specific runtime
> (`directml-inference`). Embeddings and reranking are the primary
> supported release use cases. Decoder models are **not** part of the
> first Maven Central core release; `directml-inference` is sidecar-only
> for `0.1.0-beta.1`. When no Phi-3 model directory is present the
> sidecar still starts cleanly and `summarize` replies with
> `-32005 Not implemented` instead of crashing. See
> [`SUPPORTED_MODELS.md`](SUPPORTED_MODELS.md) for the model matrix and
> [`BENCHMARK.md`](BENCHMARK.md) for the reproducible embedding
> throughput matrix (CPU + DirectML, `embed` and `embedBatch`).

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

`embed` is backed by a generic BERT-style encoder runtime
(`encoder.bert.*`) that today supports two model families behind the
same `EmbeddingModel` API:

* **`minilm`** – `sentence-transformers/all-MiniLM-L6-v2`. Produces
  384-dim, L2-normalised vectors.
* **`e5`** – `intfloat/e5-small-v2` / `e5-base-v2` / `e5-large-v2`
  and `danielheinz/e5-base-sts-en-de` (English/German fine-tune).
  Produces 384/768/1024-dim, L2-normalised vectors depending on the
  variant. Uses the conventional `"query: "` / `"passage: "` input
  prefixes (`E5Prefixes.QUERY`, `E5Prefixes.PASSAGE`).

For each family both backends exist behind the same API:

* **`DirectMl…Encoder`** – the intended product path. Hybrid CPU/GPU
  pipeline:
  * **CPU**: tokenization + word/position/tokenType embedding lookup
    (small int→float gather, no benefit from GPU dispatch).
  * **DirectML**: embedding LayerNorm + the full BERT encoder
    (Q/K/V/Linear + composite/native GELU + LayerNorm, head-layout
    reshuffles, attention, MLP, residuals) + mean-pool over
    `attentionMask` + L2-Normalize, all fused FL 1.0 / FL 2.0 kernels.
* **`Cpu…Encoder`** – pure-Java CPU backend. Supported as a local
  fallback and for smaller local workloads without a DirectML-capable GPU.
  DirectML is the intended acceleration path on Windows, but CPU-only
  usage is fully supported.

Fetch the models once. Size depends on the selected variant
(MiniLM ≈ 90 MB; E5 small ≈ 130 MB, base ≈ 440 MB, large ≈ 1.3 GB):

```powershell
pwsh scripts/download-minilm.ps1                          # MiniLM (default 384-dim)
pwsh scripts/download-e5.ps1 -Variant base-sts-en-de      # E5 de/en STS (768-dim)
pwsh scripts/download-e5.ps1 -Variant small-v2            # E5 small (384-dim)
pwsh scripts/download-e5.ps1 -Variant base-v2             # E5 base (768-dim)
pwsh scripts/download-e5.ps1 -Variant large-v2            # E5 large (1024-dim)
```

`embed("…")` then returns a real L2-normalised vector regardless of the
active family or backend.

### Family and backend selection

The sidecar picks the family via `-Dembed.model` and the backend via
`-Dembed.backend`:

```powershell
# MiniLM (default family)
./gradlew.bat :directml-sidecar:run -Dembed.backend=auto        # default: try DirectML, fall back to CPU
./gradlew.bat :directml-sidecar:run -Dembed.backend=directml    # force DirectMlMiniLmEncoder; exit 3 if unavailable
./gradlew.bat :directml-sidecar:run -Dembed.backend=cpu         # force CpuMiniLmEncoder

# E5 (variant must match the on-disk config.json)
./gradlew.bat :directml-sidecar:run `
    -Dembed.model=e5 `
    -De5.model=base-sts-en-de `
    -De5.modelDir=model/e5-base-sts-en-de `
    -Dembed.backend=auto
```

`-Dembed.model` accepts `minilm` (default) or `e5`. `-De5.model`
accepts `small-v2`, `base-v2`, `large-v2`, or `base-sts-en-de`
(default). Unknown values for either property fail visibly with
exit code `2`. For E5, `config.json` is required in the model
directory and is verified against the chosen variant
(`hidden_size`, `num_hidden_layers`, `num_attention_heads`,
`intermediate_size`, `vocab_size`, `type_vocab_size`); a mismatch
is a hard error, never a silent reshape.

A forced backend (`cpu` or `directml`) exits with code `3` if its
encoder cannot be loaded – including the case where no model directory
is present at all. `auto` falls back to CPU and records the
DirectML error message in `health.lastError`, so the workbench / clients
can see why the CPU backend was chosen. The active backend is reported
in the `health` response as `embeddingBackend` (`cpu`, `directml`,
`none`, or `error`) together with `embeddingReady`.

## Reranking (`rerank`)

The sidecar additionally exposes a **cross-encoder reranker** that scores
`(query, document)` pairs jointly through the same generic BERT compute
graph used for embeddings. Typical workflow:

1. Retrieve a candidate set (e.g. top-100 via the bi-encoder `embed` +
   a vector store on the host side).
2. Send `(query, documents, topN)` to `rerank` – the sidecar runs the
   cross-encoder on every pair, sorts by raw classifier logit and
   returns the top-N.

Configuration:

```text
-Drerank.modelDir=<path>            # override the model directory
-Drerank.backend=auto|directml|cpu  # default auto (DirectML with CPU fallback)
```

The default model location is `model/cross-encoder-ms-marco-MiniLM-L-6-v2/`.
A typical checkpoint of that family is ~90 MB. As with the embedding
encoders the rerank handler stays in `-32005 Not implemented` mode when
no model directory is present – the rest of the sidecar continues to
work normally.

See `directml-sidecar/PROTOCOL.md` for the full request/response shape.

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
    into a single FL-1.0 dispatch. Validated by `DirectMlMeanPoolKernelTest` and the existing parity test
    against the CPU encoder (`DirectMlMiniLmEmbeddingReferenceTest`).
14. ✅ L2-Normalize on DirectML (`DirectMlL2NormalizeKernel`, FL-1.0 composite) – sum-of-squares via
    `DML_OPERATOR_GEMM` (`A=x[1,N] · Bᵀ=x[N,1] → s[1,1]`), norm via `DML_OPERATOR_ELEMENT_WISE_SQRT` with
    `ε²` folded into `DML_SCALE_BIAS` so `n = sqrt(s + ε²)` is a single dispatch, then a per-lane
    `DML_OPERATOR_ELEMENT_WISE_DIVIDE` with broadcast strides `[0,0,0,0]` on the scalar norm. The encoder
    now downloads `[H]` already-normalised floats per inference instead of `[B,H]` un-normalised –
    ≈ 256× less PCIe traffic on bucket `S=256, H=384` and no CPU L2 tail. Reference test still reports
    `cos(CpuMiniLmEncoder, DirectMlMiniLmEncoder) = 1.000000` on Windows 11 in-box FL 5.0; standalone
    `DirectMlL2NormalizeKernelTest` validates parity against the CPU reference at ≤ 1e-5 and the unit-norm
    property of the output.
15. ✅ Encoder pipeline generalised for BERT-style models (`encoder.bert.*`) – introduces
    `BertEncoderConfig` (hidden/numLayers/numHeads/intermediate/maxPos/typeVocab/vocab/LN-eps/hiddenAct/
    outputDimension/poolingStrategy/normalize, with `headDim()` derivation and `validate()`),
    `BertGpuLayerWeights` (16 GpuBuffer record consumed by the layer block), `BertEmbeddingLookup` (CPU
    word+pos+tokenType gather → packed `[B,H]`), `BertPoolingWeights` (mean weights + additive mask),
    `DirectMlBertEncoderLayerBlock` and `DirectMlBertEncoderStack` (model-agnostic Q/K/V → attention →
    Wo+LN → MLP+GELU+LN pipeline). `DirectMlMiniLmEncoder` now drives the generic stack with a
    MiniLM-specific adapter (`MiniLmConfigs.toBertConfig`) and a single upload step that converts the
    MiniLM safetensors into `BertGpuLayerWeights`; the hot dispatch path no longer references any
    MiniLM-specific class. The original `DirectMlMiniLmLayerBlock` / `DirectMlMiniLmEncoderStack` survive
    as their own reference implementations so the existing standalone kernel-parity tests keep passing.
    112 / 0 fail / 3 skip; reference test still reports `cos(CpuMiniLmEncoder, DirectMlMiniLmEncoder)
    = 1.000000` on Windows 11 in-box FL 5.0.
16. ✅ E5 encoder on the generic stack – introduces a model-agnostic CPU+DirectML driver
    (`encoder.bert.BertCpuLayerWeights`, `BertCpuEncoderWeights` with a generic safetensors loader,
    `CpuBertEncoder` shared math, `DirectMlBertEncoder` shared GPU orchestrator) and an `encoder.e5`
    family adapter (`E5EncoderConfig` factories for `e5-small-v2` / `e5-base-v2` / `e5-large-v2` /
    `e5-base-sts-en-de`, `E5Prefixes` with `Role.QUERY`/`Role.PASSAGE` for the conventional
    `"query: "` / `"passage: "` inputs, and `E5Encoders.loadCpu`/`loadDirectMl` convenience
    loaders that wire safetensors + WordPiece tokenizer + the matching `BertEncoderConfig` into
    the generic driver). The DirectML compute graph (Q/K/V → attention → Wo+LN → MLP+GELU+LN
    → mean-pool → L2) is shared verbatim with MiniLM – proving the Sprint-1 abstraction holds
    on a non-MiniLM shape (12 layers / hidden 768 / 12 heads / inter 3072). The sidecar now
    accepts `-Dembed.model=minilm|e5` (default `minilm`) and `-De5.modelDir` to pick the
    embedding family, with model auto-discovery under `model/e5-base-sts-en-de/` and
    related paths. Tests: 117 / 0 fail / 3 skip; the new `E5SyntheticParityTest` validates
    CPU↔DirectML cosine > 0.999 on a synthetic E5 shape (2 layers / hidden 24 / heads 4 /
    inter 48) on real Windows 11 in-box FL 5.0 DirectML, plus a separate test that verifies
    the `query:` / `passage:` prefixes flow through the request pipeline.
17. ✅ E5 real-model hardening – `E5Variant` enum with strict
    `-De5.model=small-v2|base-v2|large-v2|base-sts-en-de` parser
    (exit code `2` on unknown), `BertConfigJson` reader for the
    on-disk `config.json` and `BertConfigJson.verifyMatches` that
    rejects any mismatch between the requested variant and the
    on-disk shape (`hidden_size`, `num_hidden_layers`,
    `num_attention_heads`, `intermediate_size`, `vocab_size`,
    `type_vocab_size`). `BertEncoderConfig.validate()` is now strict
    on the runtime invariants (`hiddenAct=gelu`, `poolingStrategy=MEAN`,
    `outputDimension=hiddenSize`). `E5Encoders.resolveConfig` requires
    `config.json` to be present – no silent "trust the variant"
    fallback. The sidecar wires the full pipeline:
    `-Dembed.model=minilm|e5` (unknown ⇒ exit `2`), `-De5.model=…`
    (unknown ⇒ exit `2`), `-De5.modelDir=…`. The Java-8 client
    (`SidecarClientConfig.embedModel` / `e5Variant` /
    `e5ModelDirectory`, propagated by `SidecarProcess.buildCommandLine`)
    and the Workbench expose the same three knobs. A new
    `E5RealModelReferenceTest` loads a real E5 checkpoint when
    present (`-De5.testModelDir` or any of the variant directory
    hints), validates `cos(CPU, DirectML) > 0.99` across an EN/DE
    corpus, asserts the `query:` / `passage:` prefixes propagate
    consistently CPU↔DirectML, and skips cleanly on machines without
    Windows/D3D12 or without a downloaded model. `scripts/download-e5.ps1`
    drives the four supported variants. Tests: 138 / 0 fail / 9 skip.
18. ✅ Reranker (cross-encoder) support – `Reranker` interface with CPU
    (`CpuReranker`) and DirectML (`DirectMlReranker`) backends built on the
    generic BERT stack; classifier head on CPU, embedding-lookup + BERT layers
    on the selected backend. `WordPieceTokenizer.encodePair(…)` handles the
    query/document pair with `[CLS] q [SEP] d [SEP]` framing.
    `BertCrossEncoderRerankers.loadCpu/loadDirectMl` wires the loader,
    `RerankHandler` exposes `rerank` over JSON-RPC, and `health` reports
    `rerankerBackend` / `rerankerReady` / `rerankerModel`. The sidecar
    accepts `-Drerank.modelDir` and `-Drerank.backend=auto|directml|cpu`
    with the same forced-mode semantics as `embed.backend` (unknown value →
    exit 2; forced mode with missing or broken model → exit 3). The Java-8
    `SidecarClient.rerank(query, documents, topN)` is covered by a
    fake-process round-trip test, the Workbench has a Rerank tab plus
    dedicated `Reranker directory` / `rerank.backend` fields in the config
    panel, and `scripts/download-reranker.ps1` drives the default
    `cross-encoder/ms-marco-MiniLM-L-6-v2` checkpoint.
    `RerankerRealModelReferenceTest` boots both backends on that real
    checkpoint and locks in CPU↔DirectML score parity (|Δ| < 1e-2) and
    identical top-N ranking order; it auto-skips when no reranker model
    is present locally.
19. ⏳ SentencePiece-BPE tokenizer for XLM-R-based multilingual E5 variants
    (`multilingual-e5-large-instruct`, JinaBERT v3, …).

Issue backlog: [`win-directml-java-issues.md`](win-directml-java-issues.md).

## Releases / Maven Central

Published artefacts under `com.aresstack:` (Sonatype Central Portal):

```gradle
dependencies {
    implementation 'com.aresstack:directml-config:0.1.0-beta.1'
    implementation 'com.aresstack:directml-windows-bindings:0.1.0-beta.1'
    implementation 'com.aresstack:directml-encoder:0.1.0-beta.1'
    // Pure-Java-8 client for the sidecar (separate JVM):
    implementation 'com.aresstack:directml-sidecar-client-java8:0.1.0-beta.1'
}
```

```xml
<dependency>
    <groupId>com.aresstack</groupId>
    <artifactId>directml-encoder</artifactId>
    <version>0.1.0-beta.1</version>
</dependency>
```

Module overview (see [`SUPPORTED_MODELS.md`](SUPPORTED_MODELS.md) and
[`MODEL_LICENSES.md`](MODEL_LICENSES.md) for what runs on what):
| Coordinate | Java | Purpose |
|---|---|---|
| `com.aresstack:directml-config` | 21 | Backend / feature-level / GPU-adapter configuration types. |
| `com.aresstack:directml-windows-bindings` | 21 (preview) | FFM bindings to D3D12 / DXGI / DirectML + low-level kernel
primitives. |
| `com.aresstack:directml-encoder` | 21 (preview) | MiniLM, E5, cross-encoder reranker pipelines with CPU + DirectML
parity. |
| `com.aresstack:directml-sidecar-client-java8` | 8 | JSON-RPC client to talk to the sidecar from a Java-8 host JVM. |
The `directml-sidecar` and `directml-sidecar-workbench` applications are **not** on Maven Central;
they are shipped as self-contained distribution zips via the
[GitHub Releases](https://github.com/aresstack/win-directml-java/releases)
page on every `v*` tag.

### Releasing a new version

Locally:

```powershell
# 1. Make sure the suite is green.
./gradlew :directml-windows-bindings:test :directml-encoder:test `
          :directml-sidecar-client-java8:test :directml-sidecar:test
# 2. Cut and push the tag (updates README, commits, tags, pushes).
.\release.ps1 0.1.0-beta.1
```

The `.github/workflows/release.yml` workflow then runs

```
./gradlew -Pversion=$Version -x test publishAggregationToCentralPortal
```

against the Sonatype Central Portal (autoPublish=true). The sidecar
and workbench distribution zips are attached to the matching GitHub Release.
Required GitHub-Actions secrets on the `aresstack/win-directml-java` repo:
| Secret | Source |
|---|---|
| `CENTRAL_USERNAME` | Sonatype Central Portal user-token |
| `CENTRAL_PASSWORD` | Sonatype Central Portal password-token |
| `GPG_PRIVATE_KEY` | `gpg --armor --export-secret-keys <KEY_ID>` |
| `GPG_PASSPHRASE` | passphrase of the above key |
The publishing configuration itself lives in the root `build.gradle`
(plugin `com.gradleup.nmcp` 0.1.5) and is applied to every module
listed under `ext.publishableModules`.
