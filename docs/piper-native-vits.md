# Native Piper VITS in win-directml-java

## Goal and architecture

Support Piper text-to-speech in the project's **own Java/DirectML engine**, using
ONNX strictly as a **model import format**, as already done for Phi-3.
There is **no ONNX Runtime or Python dependency** in the inference path.

The first milestone is now a **complete CPU reference implementation**.
`PiperNativeGraphPrefixRunner` executes all **2,682/2,682** operators of the
tested German Piper voice and produces a real waveform. The implementation is
deliberately still called a **reference**: no DirectML GPU backend or native
eSpeak text phonemizer has been integrated, and audible quality/parity against
upstream Piper has not yet been established.

## Verified German model

- Source: `rhasspy/piper-voices` / `de/de_DE/thorsten/medium`
- Files: `de_DE-thorsten-medium.onnx` (~63 MB) and the paired `.onnx.json`
- Inputs: `input` (INT64 phoneme IDs), `input_lengths` (INT64),
  `scales` (FLOAT32: noise, speaking speed, duration noise)
- Output: mono floating point audio, sampled at **22,050 Hz**
- Graph: **2,682 nodes**, 405 initializers, including 129 `Conv`
  and 3 `ConvTranspose` operations
- All nodes execute using the repository's own Java code, not ONNX Runtime.
- The opt-in model test checks that the produced audio is non-silent, finite
  and bounded by the final tanh output.

Do not commit voice/model binaries. `.gitignore` excludes `model/piper-*/`.
Check the particular voice/dataset license before redistributing a voice.

## Implemented Java components

- **Voice and phonemes:** `PiperVoiceConfig`,
  `PiperPhonemeIdEncoder`, `PiperSynthesisInput`,
  and the replaceable `PiperPhonemizer` interface.
  `EspeakNgCliPhonemizer` is an optional adapter for a separately installed
  eSpeak-NG executable. It invokes `--ipa=1 --stdin` with UTF-8 text and
  bounded execution time; no eSpeak binary is distributed with the JAR.
  The adapter has isolated command tests, but has **not yet passed a real
  eSpeak-NG IPA/voice parity test** on this machine.
- **Model source:** `PiperOnnxModelSource` imports the ONNX protobuf
  through `OnnxModelReader`; the reader was extended to preserve tensor
  constants and handle ONNX attribute scalar fields correctly.
- **Native reference execution:** `PiperNativeGraphPrefixRunner` evaluates
  the entire graph. `PiperTensor` carries typed shapes and data;
  `PiperTensorOperations`, `PiperIndexOperations`,
  `PiperShapeOperations`, `PiperReductionOperations`, and
  `PiperActivationOperations` implement the required computation.
- **Audio decoder:** `PiperConvolutionExecutor` adapts ONNX NCL tensor
  layouts to the independent grouped `PiperConv1dReference` and
  `PiperConvTranspose1dReference` kernels.
- **Synthesis API:** `PiperReferenceSynthesizer.synthesizePhonemes(ipa)`
  runs the complete model and returns immutable `PiperAudio`.
  `synthesizeText(text, phonemizer)` additionally uses an injected
  `PiperPhonemizer` strategy; a full eSpeak-NG implementation is not yet
  bundled. `PiperWavWriter` exports mono PCM16 WAV.

The `RandomNormalLike` reference currently uses an explicitly seeded
`java.util.Random`. Its statistical contract is implemented, but its
random samples are **not bit-exact** with ONNX or upstream Piper. Deterministic
integration tests use seed 42.

## Local smoke test

Place the two voice files in `model/piper-de_DE-thorsten-medium/`. Run:

```powershell
cmd.exe /c ".\gradlew.bat :directml-inference:test --tests com.aresstack.windirectml.inference.piper.* -Dpiper.realModel=true --no-daemon --max-workers=1"
```

This includes `PiperReferenceSynthesizerTest`, which synthesizes the test
IPA string `hˈaloː` and writes a WAV for manual listening:

```
directml-inference/build/piper-demo/thorsten-hallo.wav
```

The WAV file remains in the ignored build directory. **A successful WAV
export is not a passed perceptual listening test.** It verifies the complete
numeric graph and produces audio samples, but pronunciation and acoustic
parity still need to be reviewed.

Normal unit tests (no model file required):

```powershell
.\gradlew.bat :directml-inference:test --tests 'com.aresstack.windirectml.inference.piper.*' --no-daemon --max-workers=1
```

The optional real-model suite is disabled on standard CI and other platforms.

## Remaining work

1. **Text phonemization:** Verify the optional eSpeak-NG CLI adapter against a
   real installed phonemizer and Piper's native eSpeak phoneme sequences,
   including UTF-8, sentence splitting, language selection, IPA mapping,
   punctuation and voice-specific pronunciation parity. The CLI adapter
   remains optional, with no bundled GPL binaries.
2. **DirectML GPU compiler:** Reuse the Phi-3 ONNX-to-`.wdmlpack` architecture.
   Lower VITS tensor layouts and all required operators to the existing DML
   device/runtime; validate dynamic shapes, memory reuse, GPU dispatch
   ordering and random sampling behavior. The working CPU reference is the
   numerical oracle, not a performance substitute.
3. **Parity and listening:** Compare real-voice waveform duration and
   acoustic quality with the upstream Piper implementation, taking the random
   seed into account. Run a human listening test for German pronunciation.
4. **Production performance:** Avoid keeping all intermediate graph outputs
   live, reuse model weights and compiled operators, process long text by
   sentence, stream output and add cancellation.
5. **Licenses and release:** Check the model-specific voice/dataset license
   and eSpeak-NG license before bundling weights or phonemizer binaries.
   Only promote the API to production when those checks and GPU acceptance
   tests pass.

## Sources

- Model and license metadata:
  https://huggingface.co/rhasspy/piper-voices/tree/main/de/de_DE/thorsten/medium
- Piper voice and phoneme format:
  https://github.com/rhasspy/piper
- eSpeak-NG:
  https://github.com/espeak-ng/espeak-ng
