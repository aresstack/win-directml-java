# win-directml-java

Local **embeddings and reranking** for Java 21 applications that want to reduce
or replace cloud inference calls. The core product is a pure Java API for a
Windows CPU / DirectML inference engine. It can be embedded directly in another
Java 21 application, such as an ACP sidecar or manager-agent project, without
starting this repository's legacy JSON-RPC sidecar process.

The supported release use case is a local RAG / search pipeline:

```text
Java 21 application / ACP sidecar / manager project
  -> directml-runtime API
  -> local embeddings
  -> your vector index
  -> local reranking
  -> sensitive text stays on-prem/offline
```

Use this when you want:

- **Data control:** document text and search queries do not have to leave the
  machine or network boundary.
- **Predictable cost:** embedding and reranking calls are local compute instead
  of per-token/per-request cloud billing.
- **Offline / on-prem operation:** CPU-only operation is supported; DirectML is
  an acceleration path, not a requirement.
- **Java 21 integration:** the inference engine is consumed as a normal Java
  library. A sidecar/container/process wrapper is optional and belongs to the
  consuming application architecture.

ACP/MCP/A2A orchestration and manager-agent logic are intentionally not part of
this repository. Those belong in a separate ACP sidecar / manager project that
consumes this library as a local ML capability.

Decoder/seq2seq **text generation** and **Whisper speech transcription**
are implemented in the separate `directml-inference` module. Its APIs are
experimental and this module is configured for Maven publication. The
**Workbench** application is not a published artifact; it provides a UI
for text generation and a separate Whisper microphone demo. See
[Text generation in the Workbench](#text-generation-in-the-workbench-experimental)
and [Speech transcription with Whisper](#speech-transcription-with-whisper-experimental).
## Recommended model choices

| Need                                          | Recommended path                         | Notes                                                                                     |
|-----------------------------------------------|------------------------------------------|-------------------------------------------------------------------------------------------|
| Fast default embeddings                       | `sentence-transformers/all-MiniLM-L6-v2` | Small, quick, good release baseline.                                                      |
| Higher-quality E5-style retrieval             | `intfloat/e5-base-v2`                    | Use `query: ` for queries and `passage: ` for indexed documents.                          |
| Maximum E5 capacity in current WordPiece path | `intfloat/e5-large-v2`                   | Experimental and much larger.                                                             |
| Better final ordering after vector search     | `cross-encoder/ms-marco-MiniLM-L-6-v2`   | Scores query/document pairs jointly; best used on top-K candidates.                       |
| German E5 STS checkpoint                      | `danielheinz/e5-base-sts-en-de`          | Planned only: current upstream is XLM-R/SentencePiece, not the shipped WordPiece E5 path. |

See [`SUPPORTED_MODELS.md`](SUPPORTED_MODELS.md) for the full support matrix
and [`MODEL_LICENSES.md`](MODEL_LICENSES.md) for model license notes.

## CPU vs. DirectML guidance

`cpu`, `directml` and `auto` are first-class backend choices.

| Mode               | When to use it                                               | Behavior                                             |
|--------------------|--------------------------------------------------------------|------------------------------------------------------|
| `Backend.CPU`      | Servers, CI, VMs or desktops without a DirectML-capable GPU  | Pure local CPU path. Not a failure mode.             |
| `Backend.DIRECTML` | You require GPU acceleration and want failures to be visible | Fails fast if DirectML/model loading is unavailable. |
| `Backend.AUTO`     | Default application mode                                     | Tries DirectML first and falls back to CPU.          |

DirectML improves latency and throughput on Windows machines with suitable GPUs.
CPU remains useful for small workloads, tests, offline environments and systems
where GPU acceleration is not available. See [`BENCHMARK.md`](BENCHMARK.md)
for reproducible throughput numbers and [`docs/fallback-policy.md`](docs/fallback-policy.md)
for the backend contract.

## Typical local RAG pipeline

1. **Chunk documents** in the host application.
2. **Embed chunks locally** with MiniLM or E5.
3. **Store vectors** in your own vector index or database.
4. **Search top-K** by vector similarity.
5. **Rerank top-K locally** with the cross-encoder reranker.
6. Optionally pass the final context to a later local or remote summarization
   layer owned by your application.

For E5, prefix query texts with `query: ` and indexed/passages with `passage: `.
Reranker scores are raw model logits: compare them only within the same query,
not across different queries or models. The `rerank` result is sorted descending
by score.

## Direct Java 21 API

Java 21 applications can use embeddings and reranking directly in-process:

```java
import com.aresstack.windirectml.encoder.e5.E5Variant;
import com.aresstack.windirectml.runtime.facade.*;

// Create runtime (defaults to auto backend: DirectML → CPU fallback)
LocalMlRuntime runtime = LocalMlRuntime.create();

// Load and use embedding model (MiniLM)
var embedCfg = EmbeddingModelConfig.miniLm(Path.of("model/all-MiniLM-L6-v2"));
try (var embeddings = runtime.loadEmbeddingModel(embedCfg)) {
    float[] vector = embeddings.embed("hello world");
    List<float[]> batch = embeddings.embedBatch(List.of("a", "b", "c"));
}

// E5 requires an explicit WordPiece variant (SMALL_V2, BASE_V2, LARGE_V2)
var e5Cfg = EmbeddingModelConfig.e5(Path.of("model/e5-base-v2"), E5Variant.BASE_V2, "query: ");
try (var embeddings = runtime.loadEmbeddingModel(e5Cfg)) {
    float[] vector = embeddings.embed("search query");
}

// Load and use reranker
var rerankCfg = new RerankerModelConfig(Path.of("model/cross-encoder-ms-marco-MiniLM-L-6-v2"));
try (var reranker = runtime.loadRerankerModel(rerankCfg)) {
    var results = reranker.rerank("search query", List.of("doc1", "doc2"));
}
```

Dependency (Gradle):

```groovy
implementation 'com.aresstack:directml-runtime:0.3.0'
```

The API hides transport details and does not require JSON-RPC or a separate
process. Supported WordPiece E5 variants: `small-v2`, `base-v2`, `large-v2`.
XLM-R/SentencePiece E5 models (e.g. `danielheinz/e5-base-sts-en-de`) remain
planned but not ready; attempting to load them throws `UnsupportedModelException`.

## Speech transcription with Whisper (experimental)

`directml-inference` provides an **in-process Java 21 speech-to-text API** for
the multilingual [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny)
checkpoint. Whisper uses Java audio processing and a **CPU reference**
encoder/decoder. It does not use Whisper.cpp, Python, ONNX Runtime, or custom
native DLLs. Whisper **does not yet run on DirectML/GPU**.

### Java API

```java
import com.aresstack.windirectml.inference.whisper.WhisperTranscriber;

import java.io.IOException;
import java.nio.file.Path;

public final class WhisperExample {
    public static void main(String[] args) throws IOException {
        WhisperTranscriber transcriber =
                new WhisperTranscriber(Path.of("model", "whisper-tiny"));

        String transcript = transcriber.transcribe(
                Path.of("recordings", "german-speech.wav"), 100);
        System.out.println(transcript);
    }
}
```

The constructor takes the **model directory**. Reuse the transcriber rather
than reloading model weights for each recording. Other public entry points:

- `transcribe(float[] samples, int maxTokensPerChunk)`: process **16 kHz mono
  float PCM** in 30-second chunks.
- `transcribeShortRecording(float[] samples, int maxTokensPerChunk)`: process
  short microphone recordings in smaller six-second chunks.
- `transcribeMel(float[][] features, int maxTokens)`: process an already
  Whisper-compatible log-Mel spectrogram (lower-level API).

These original methods return a synchronous `String` containing recognized speech. Input
`float[]` samples are normalized, mono, 16,000 Hz PCM (nominally -1 to +1);
raw PCM16 byte arrays must first be decoded and resampled. The WAV overload
performs WAV decoding, mono conversion and resampling for you. The integer
argument limits generated tokens per chunk.

### Segment timestamps (planned patch 0.3.1)

Whisper Tiny can emit 20-ms timestamp tokens. The additive 0.3.1 API exposes
only intervals that the model emitted, without inventing segment boundaries:

```java
WhisperTranscriber transcriber = new WhisperTranscriber(Path.of("model", "whisper-tiny"));
TranscriptionResult result = transcriber.transcribeWithTimestamps(
        Path.of("recordings", "german-speech.wav"), 400);
for (TranscriptionSegment segment : result.getSegments()) {
    System.out.println(segment.getStartMillis() + " - "
            + segment.getEndMillis() + ": " + segment.getText());
}
```

Import `TranscriptionResult` and `TranscriptionSegment` from
`com.aresstack.windirectml.inference.whisper`. For an existing 16-kHz mono
`float[]` use `transcribeWithTimestamps(float[], int)` or
`transcribeShortRecordingWithTimestamps(float[], int)` for microphone recordings.
Each result contains `getText()` for the full transcript, `getSegments()` for
model-confirmed start/end positions in milliseconds and `getUnalignedText()`
for words without a complete timestamp pair. Segment alignment is experimental;
word-level times and streaming are not provided. Audio uses 30-second
Whisper windows; all confirmed segments except the last are retained, and
the next window starts at the last segment's absolute start time. The last
window retains all segments. If timestamps cannot safely advance processing,
the full window is retained and processing advances by 30 seconds instead.
No separate pause-based segmentation, VAD or NLP is applied. The original
plain-text API is unchanged. Use a sufficient token limit (for example, 400
for longer recordings) to leave room for the timestamp tokens.

### Download model weights

Download the **multilingual** model `openai/whisper-tiny` (not `.en`) and
place the following files together under `model/whisper-tiny/`:

- `config.json`
- `model.safetensors`
- `vocab.json`
- `tokenizer.json`

Model weights are neither committed to this repository nor bundled in the
JAR. `model/whisper-*/` is git-ignored. Review the model's upstream license
before distributing the weights.

### Local microphone trial

Run the Java main class
`com.aresstack.windirectml.workbench.whisper.WhisperMicrophoneDemo` from
IntelliJ or launch the Gradle task:

```bash
bash ./gradlew :directml-workbench:whisperMicrophoneDemo
```

On the Windows development machine there is also
`tools/run-whisper-microphone.cmd`. The separate Swing window lets you choose
a microphone, start and stop recording, then displays the transcribed German
text. Enable **Zeitstempel anzeigen (experimentell)** in the demo to display
`[MM:SS.cc – MM:SS.cc]` before each recognized segment. Disable it to compare
with the original plain-text decoding. The demo records until stopped manually so that longer recordings
and minute-level offsets can be tested. Unaligned speech is displayed
separately, not assigned a guessed time. Longer recordings can take several minutes to process on the CPU; the UI
remains responsive while decoding runs in the background. The window is a **Workbench demo**, not part of
the library API.

### Tests, limitations and Maven Central

Real-checkpoint tests validate the encoder/decoder, preprocessing, WAV input
and the short-recording path. A synthesized German speech sample was
recognized as "Hallo? Dies ist ein Test der Deutschen Spracherkennung."
(the original sentence differs slightly in case and punctuation).

Whisper transcription is currently **experimental**, synchronous and
CPU-bound. Run it off the UI thread. Incremental streaming results,
speaker identification and word-level timestamps are not implemented. The
0.3.1 timestamp API offers model-confirmed segment times only; its greedy
decoder may occasionally leave passages without a complete time interval.
Accuracy for longer, varied speech recordings remains to be established by
real-world testing.

Version **0.3.0** introduces the experimental Whisper API in the published
`com.aresstack:directml-inference:0.3.0` artifact. Earlier versions, including
0.2.1, do not contain the Whisper transcription API. Segment timestamps are
being added for a separate 0.3.1 patch release (not part of 0.3.0).

## Relationship to sidecars and ACP

This repository is a Windows inference library. It deliberately does not contain
ACP, MCP, A2A, graph routing or manager-agent orchestration.

A future ACP sidecar or manager-agent project should depend on `directml-runtime`
and call the Java 21 API directly:

```text
ACP / manager project
  -> directml-runtime
  -> directml-encoder
  -> directml-windows-bindings
  -> Windows CPU / DirectML
```

The older JSON-RPC sidecar, Java 8 client, protocol module and workbench remain
in the source tree as beta-era compatibility code. They are no longer the primary
product path and are not part of new Maven Central releases unless explicitly
reactivated later.

## Modules

```text
directml-config                     Minimal inference configuration and registry types
directml-windows-bindings           Java 21 FFM bindings for DXGI, D3D12, DirectML, COM/HRESULT
directml-encoder                    MiniLM, E5 and reranker encoder/runtime code
directml-runtime                    Public Java 21 API for direct in-process use

Additional Java 21 inference module (configured for publication; experimental APIs):
directml-inference                  Native Java inference runtime + Whisper transcription + wdmlpack compilers/loaders
                                    (Gemma, Qwen, SmolLM2, T5/CodeT5, Phi-3); no Python, no ONNX Runtime
Workbench applications (not published in new Maven releases):
directml-workbench                  Java 21 Swing Workbench (Download/Convert, generation + microphone demo)
directml-workbench-launcher         Launcher for the Java 21 Workbench

Legacy / beta-era modules kept in source, not published in new releases:
directml-sidecar-protocol-java8     Shared protocol/validation types for old sidecar bridge
directml-sidecar                    JSON-RPC 2.0 sidecar adapter over directml-runtime
directml-sidecar-client-java8       Java 8 client for the old JSON-RPC sidecar
directml-sidecar-workbench          Java 8 Swing workbench / diagnostics UI (legacy)
```

## Requirements

- Windows 11 for DirectML acceleration.
- JDK 21 for the runtime modules.
- Downloaded model directories; model weights are not shipped in Maven artifacts.

FFM is still a preview feature, so Java 21 runtime execution uses:

```text
--enable-preview
--enable-native-access=ALL-UNNAMED
```

## Download local models

Fetch model files once via the **Workbench Download** tab. The Workbench keeps the files under `model/` by default and offers the supported MiniLM, E5 and reranker variants through the UI.

Use **Force re-download** in the Download tab when a local model folder is incomplete or should be refreshed.

The Workbench E5 download support intentionally does **not** offer
`danielheinz/e5-base-sts-en-de`; the current upstream checkpoint is
XLM-R/SentencePiece and is tracked as planned until the SentencePiece/XLM-R
runtime path exists.

## Embeddings and reranking details

### Embeddings

Embeddings are backed by a generic BERT-style encoder runtime. The current
shipped/experimental WordPiece families are MiniLM and E5.

- `minilm`: `sentence-transformers/all-MiniLM-L6-v2`, 384-dimensional output.
- `e5`: `intfloat/e5-small-v2`, `intfloat/e5-base-v2`, experimental
  `intfloat/e5-large-v2`, with 384/768/1024-dimensional output.

`embedBatch` preserves input order. Planned XLM-R/SentencePiece models remain
visible in `SUPPORTED_MODELS.md` but are not silently treated as runnable.

### Reranking

Reranking scores `(query, document)` pairs jointly through a cross-encoder and
returns the top-N documents sorted by raw classifier logit. It is intended for
reranking a vector-search candidate set, not for indexing the entire corpus.

The default model location in examples is
`model/cross-encoder-ms-marco-MiniLM-L-6-v2/`.

## Text generation in the Workbench (experimental)

Separate from the published embeddings/reranking library, the repository ships a Windows
**Workbench** app (`directml-workbench`, launched via `directml-workbench-launcher`) with an
experimental text-generation tab. Generation runs as a **native Java / DirectML** path
(`directml-inference`) — **no Python and no ONNX Runtime** in the Workbench runtime path. Models load
from an internal `.wdmlpack` package compiled on the **Download → Convert** step; inference never
compiles.

Backend semantics for generation:

- **WARP** — the D3D12 software rasterizer. The dense projections / matmuls run on DirectML (WARP
  adapter); norms, attention softmax, RoPE and KV-cache stay on the CPU. This is the CPU-only-friendly
  main path.
- **AUTO** — the same runtime on the first hardware adapter when one is present, otherwise the CPU
  reference path. Hardware acceleration is opportunistic; it is not uniformly faster/stable everywhere.
- **CPU** — the validated reference runtime (Gemma's `Backend=CPU` is the one legacy external
  Python/Transformers path and is clearly labelled as such).

Runnable generation models (status is internal/experimental, surfaced honestly in the Workbench):

| Model | Status | Notes |
|-------|--------|-------|
| `google/gemma-3-270m-it` | runnable | Native Java/DirectML (WARP/AUTO); `Backend=CPU` = legacy external Python. |
| `Qwen/Qwen2.5-Coder-0.5B-Instruct` | runnable | Native DirectML INT4 (`model_q4f16.wdmlpack`). 1.5B/3B planned. |
| `HuggingFaceTB/SmolLM2-135M/360M-Instruct` | runnable | Native DirectML/WARP dense projections + CPU reference fallback. |
| `google-t5/t5-small`, `google/flan-t5-small`, `Salesforce/codet5-small`, `Salesforce/codet5-base-multi-sum` | runnable + real-certified | Mixed DirectML/WARP path; all four real-certified CPU == WARP (greedy). |
| `microsoft/Phi-3-mini-4k-instruct-onnx` | runnable | From `model_phi3.wdmlpack` via a heap-light package load (CPU path). |
| `microsoft/Phi-3.5-mini-instruct-onnx` | planned | Same architecture; would reuse the Phi-3 compiler/runtime once weights are wired. |

Workbench generation quickstart:

1. Build the repo (see [Build](#build)).
2. Launch the Workbench (`directml-workbench-launcher`, e.g. `./gradlew :directml-workbench-launcher:run`).
3. **Download** tab: download a model (e.g. Phi-3-mini or a T5 model).
4. **Convert** the download to its `.wdmlpack` from the Download tab's package action.
5. **Summarizer** tab: select the model, choose a Backend (WARP / AUTO / CPU), enter text, run.

Known limits: text generation is experimental; the Workbench app is not
published, while `directml-inference` is configured for Maven publication. Phi-3.5 stays
planned; larger Qwen (1.5B/3B) stay planned. The T5 certification covers exactly the four curated
models above — other T5/CodeT5 checkpoints are not blanket-certified. Real-model tests are opt-in;
large models can need several GB of RAM; downloaded `model/...` artifacts are git-ignored and never
committed. See [`SUPPORTED_MODELS.md`](SUPPORTED_MODELS.md), [`docs/workbench-model-status.md`](docs/workbench-model-status.md)
and [`docs/release-notes-next.md`](docs/release-notes-next.md).

## Status summary

| Area                                    | Status                                                       |
|-----------------------------------------|--------------------------------------------------------------|
| Direct Java 21 API                      | Primary product path, implemented in `directml-runtime`.     |
| MiniLM embeddings                       | CPU and DirectML paths implemented.                          |
| E5 WordPiece embeddings                 | `small-v2`, `base-v2`, experimental `large-v2` runtime path. |
| Reranker                                | Cross-encoder reranker available through the direct API.     |
| CPU-only usage                          | Supported; not a failure mode.                               |
| DirectML usage                          | Supported Windows acceleration path.                         |
| Whisper-Tiny speech transcription         | Experimental Java 21 API in `directml-inference` since 0.3.0; German real-model test passed, CPU only. |
| Legacy JSON-RPC sidecar / Java 8 bridge | Source kept, no longer published in new releases.            |
| Workbench text generation               | Experimental Workbench app feature (not a published artifact): Gemma-it, Qwen 0.5B, SmolLM2, T5/CodeT5 (4 curated, real-certified), Phi-3-mini runnable; Phi-3.5 + larger Qwen planned. Native Java/DirectML, no Python/ONNX Runtime. |
| ACP / MCP / A2A orchestration           | Out of scope for this repository.                            |

## Build

### GPT-compatible release ZIP

For GPT or other constrained execution environments, use the rolling `gpt-compatible` GitHub release. It contains the
repository plus a prepared `.chatgpt/gradle-home` cache so Gradle can run offline after extraction.

From the extracted release ZIP, run:

```bash
bash chatgpt-build.sh
```

Details are documented in [`CHATGPT_BUILD.md`](CHATGPT_BUILD.md).

PowerShell:

```powershell
$env:JAVA_HOME = "C:\Program Files\Zulu\zulu-21"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
./gradlew.bat clean build
```

See [`howtobuild.md`](howtobuild.md) for details.

## Runtime configuration

| System property                 | Default                      | Effect                                                                                  |
|---------------------------------|------------------------------|-----------------------------------------------------------------------------------------|
| `windirectml.debug`             | `false`                      | Enables the D3D12 + DirectML debug layer and surfaces InfoQueue messages in exceptions. |
| `windirectml.directml.dll`      | unset, use in-box `System32` | Absolute path to a deliberately shipped `DirectML.dll` redistributable.                 |
| `windirectml.dxgi.adapterIndex` | `0`                          | DXGI adapter index to bind.                                                             |

DirectML operators are gated by `DML_FEATURE_LEVEL`; unsupported fused kernels
fall back to portable composite kernels or skip tests where appropriate. Do not
point `windirectml.directml.dll` at DLLs installed by unrelated software under
`C:\Program Files\...`; use the in-box DLL or an application-owned
redistributable.

## Releases / Maven Central

The Gradle Maven Central publication set contains seven Java library modules,
including `directml-inference`. The Swing Workbench and legacy sidecar remain unpublished.

```gradle
dependencies {
    implementation 'com.aresstack:directml-runtime:0.3.0'
}
```

Gradle-configured publishable artifacts under `com.aresstack:`:

| Coordinate                                | Java        | Purpose                                                                     |
|-------------------------------------------|-------------|-----------------------------------------------------------------------------|
| `com.aresstack:directml-config`           | 8/21 shared | Configuration, limits and model registry types.                             |
| `com.aresstack:directml-windows-bindings` | 21 preview  | FFM bindings to D3D12 / DXGI / DirectML plus low-level kernel primitives.   |
| `com.aresstack:directml-encoder`          | 21 preview  | MiniLM, E5 and cross-encoder reranker pipelines with CPU + DirectML parity. |
| `com.aresstack:directml-runtime`          | 21 preview  | Public direct Java 21 ML API for embeddings and reranking.                  |
| `com.aresstack:directml-model-catalog` | 8/21 shared | Neutral model catalog metadata and backend compatibility. |
| `com.aresstack:directml-model-package` | 21 preview | Model package/load contract used by inference. |
| `com.aresstack:directml-inference` | 21 preview | Experimental text generation and Java CPU Whisper transcription API. |

The `directml-inference` module is configured for publication. The Workbench
applications (`directml-workbench`, `directml-workbench-launcher`) and the
legacy beta sidecar artifacts are not published. API maturity and Maven artifact
publication status are separate: Whisper and text generation remain experimental.

Full artifact overview: [`docs/artifact-structure.md`](docs/artifact-structure.md).

### How Maven Central publishing works

**Publish all seven modules together in one deployment.** The root
`build.gradle` lists the seven artifact IDs in `publishableModules` and
configures their Maven publications, POM metadata, source/Javadoc JARs and
GPG signatures. The `com.gradleup.nmcp.aggregation` plugin combines them into
**one** Sonatype Central Portal bundle using
`:publishAggregationToCentralPortal`. The bundle contains seven components,
all with the same version; **one deployment is not one artifact**.

The GitHub Actions workflow [`release.yml`](.github/workflows/release.yml)
runs the aggregate task on a Java 21 Ubuntu runner. Sonatype validates the
uploaded components and, with `publishingType = 'AUTOMATIC'` in the root
`nmcpAggregation.centralPortal` configuration, publishes the bundle without
an extra manual Publish click. The Swing Workbench, launcher and legacy
JSON-RPC/Java-8 sidecar modules are excluded.

**Never use `publishAllPublicationsToCentralPortal` for a normal release.**
That older command uploads the modules as separate deployments; it caused the
split 0.3.0 publication and complicates error recovery. Use the **root**
`:publishAggregationToCentralPortal` task instead.

### Repository setup (one time)

Configure these GitHub Actions repository secrets in
`aresstack/win-directml-java`:

| GitHub secret | Purpose | Gradle property in `release.yml` |
|---|---|---|
| `CENTRAL_USERNAME` | Sonatype Central Portal publishing token username | `centralUsername` |
| `CENTRAL_PASSWORD` | Sonatype Central Portal publishing token password | `centralPassword` |
| `GPG_PRIVATE_KEY` | ASCII-armored PGP private key for Maven signatures | `signingInMemoryKey` |
| `GPG_PASSPHRASE` | Passphrase for that key | `signingInMemoryKeyPassword` |

The workflow passes these as `ORG_GRADLE_PROJECT_*` environment variables.
The **root** `nmcpAggregation.centralPortal` block consumes the Sonatype
credentials; the individual Maven publications consume the signing key.
Do not commit tokens or private keys. Local `publishToMavenLocal` does not
require publishing credentials.

### Publish a new version

1. Merge the intended changes into `main`. Confirm that the CI jobs for the
   final code commit are green. The publishing workflow intentionally uses
   `-x test`, so it **does not replace CI/testing**. GPU and real-model
   reference tests are separate checks when the release requires them.
2. From the repository root on Windows, update `main` and test the new
   version (replace `0.4.0` below with the next unused version):

   ```powershell
   git switch main
   git pull --ff-only origin main
   git status --short
   .\gradlew.bat clean build -Pversion=0.4.0 --no-daemon
   .\gradlew.bat publishToMavenLocal -Pversion=0.4.0 --no-daemon
   .\gradlew.bat :zipAggregation -Pversion=0.4.0 --no-daemon
   ```

   Check that `build/nmcp/zip/aggregation.zip` contains POMs for **all
   seven** modules at the new version. Building the ZIP locally does **not**
   upload it to Sonatype.
3. Create the release from a clean, up-to-date `main` branch:

   ```powershell
   .\release.ps1 0.4.0
   ```

   The script updates version references in the README, commits any changes,
   creates `v0.4.0`, and pushes the branch and tag. Pushing the **new tag**
   automatically triggers `release.yml`, which derives `0.4.0` from the
   tag. Never move or reuse a published version tag.
4. Watch the **Release to Maven Central** run under
   [GitHub Actions](https://github.com/aresstack/win-directml-java/actions),
   or inspect the runs and logs with GitHub CLI:

   ```powershell
   gh run list -R aresstack/win-directml-java --workflow release.yml --limit 5
   gh run view RUN_ID -R aresstack/win-directml-java --log
   ```

   Gradle builds and signs the seven publications, creates one aggregation
   ZIP and uploads it to Sonatype. A successful local build or upload alone
   does not mean the artifacts are publicly available.
5. Verify Sonatype's **Deployments** tab reports `PUBLISHED`, then check
   the seven `com.aresstack` coordinates on
   [Maven Central](https://repo.maven.apache.org/maven2/com/aresstack/).
   Confirm the new-version `.pom`, `.jar`, `-sources.jar` and
   `-javadoc.jar` files for each module (HTTP 200). Public availability
   can lag behind the initial Sonatype upload.

For an explicitly controlled rerun, `release.yml` also supports
`workflow_dispatch` with a `version` input and a selected branch. **Do not
run it blindly for an existing version**: Maven Central coordinates cannot
be overwritten, and an earlier deployment may still be publishing.

### Failed workflows, slow Sonatype publishing and recovery

A red GitHub Actions job is **not** proof that publication failed. For
0.3.0, Sonatype accepted individual uploads while Gradle's deployment-status
checks returned HTTP 404; a subsequent aggregate upload encountered duplicate
coordinates. The final three-component recovery upload then exceeded the
nmcp plugin's **10-minute wait timeout**, but Sonatype later marked it
`PUBLISHED`. All seven 0.3.0 artifacts became publicly available.

Follow this order when a publishing run fails:

1. Read the failed workflow log and identify the **Sonatype deployment UUID**
   from `Nmcp: deployment bundle ... uploaded`. Check whether Gradle failed
   **before upload**, during **validation**, or only while **waiting**.
2. Run the read-only
   [`inspect-sonatype.yml`](.github/workflows/inspect-sonatype.yml)
   workflow from `main` with one or more space-separated UUIDs as the
   `deployment_ids` input. For example:

   ```powershell
   gh workflow run inspect-sonatype.yml -R aresstack/win-directml-java --ref main -f "deployment_ids=<deployment-uuid>"
   ```

   Read its logs for `deploymentState` and `errors`. It does not upload
   anything. A deployment may be `PUBLISHING`, `PUBLISHED` or `FAILED`.
3. If it is still `PUBLISHING`, **wait; do not start another upload**. If
   it becomes `PUBLISHED`, verify the actual artifacts on Maven Central.
   A timeout or transient status-query error alone is not a reason to retry.
4. Only if some coordinates are genuinely **missing** after checking both
   Sonatype and Maven Central, prepare a temporary, **isolated recovery
   branch** whose `publishableModules` includes *only* missing modules.
   Inspect `build/nmcp/zip/aggregation.zip` before any upload. Never
   overwrite existing coordinates, re-tag `vX.Y.Z` or merge a
   recovery-only module list into `main`. If the desired version cannot
   safely be completed, use a new patch version after reviewing the state.

See also [`build.gradle`](build.gradle) for the exact publication set and
Sonatype plugin configuration, and [`release.ps1`](release.ps1) for the
new-version tagging procedure.
