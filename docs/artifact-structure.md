# Maven artifact structure

Starting with 0.3.1, Maven Central publishes nine library modules in one aggregated
Sonatype deployment under group `com.aresstack`. The WinDirectML Swing workbench,
launcher, and legacy JSON-RPC sidecar modules remain unpublished.

## Public artifacts

| Artifact | Java | Purpose |
|---|---|---|
| `directml-config` | 8-compatible | Shared runtime configuration |
| `directml-model-catalog` | 8-compatible | Model catalog and capabilities |
| `directml-model-package` | 21 | Model package contracts |
| `directml-windows-bindings` | 21 preview | Windows DirectML and D3D12 bindings |
| `directml-encoder` | 21 preview | Embedding and reranking encoders |
| `directml-runtime` | 21 preview | Public local inference API |
| `directml-inference` | 21 preview | Text generation and experimental Whisper transcription |
| `audio-dsp` | 8 | Ask AI's reusable PCM16 filtering and voice activity detection pipeline |
| `directml-whisper-audio` | 21 | Optional Ask AI PCM processing and VAD adapter for Whisper |

The `audio-dsp` module works independently of Whisper, DirectML, and any
neural inference runtime. The `directml-whisper-audio` module depends on the
`audio-dsp` and `directml-inference` libraries. It exposes
`com.aresstack.windirectml.whisper.audio.AskAiWhisperPipeline` for optional
speech-region detection, filtering and timestamp-preserving transcription.
The unprocessed `WhisperTranscriber` API remains available independently.

## Gradle dependencies

A Java 21 application can use the optional audio pipeline:

```groovy
dependencies {
    implementation 'com.aresstack:directml-whisper-audio:0.3.1'
}
```

Or depend only on the direct inference engine:

```groovy
dependencies {
    implementation 'com.aresstack:directml-inference:0.3.1'
}
```

For general PCM processing without Whisper:

```groovy
dependencies {
    implementation 'com.aresstack:audio-dsp:0.3.1'
}
```

## Compatibility and distribution

- Core inference and the Whisper adapter require Java 21.
- The `audio-dsp` module is compiled with Java 8 bytecode and operates on
  signed 16-bit PCM; its configured sample rate is independent of bit depth.
- The workbench is built as its own Shadow JAR and is **not** distributed
  through Maven Central.
- Model weights are never bundled into Maven Central artifacts.
- Publish the entire artifact set in one Sonatype aggregation; never release
  individual artifacts with inconsistent versions.
- Verify the CI status, local `:zipAggregation` bundle and Sonatype publication
  state before claiming a version is publicly available.
