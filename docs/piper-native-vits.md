# Piper VITS import and native Java/DirectML implementation

## Scope

Implement Piper TTS in this repository's **own** Java/DirectML engine. Use ONNX
as the **input format**, exactly as the Phi-3 import does: parse its graph,
extract and compile tensors, and execute our own kernels. **Do not add ONNX
Runtime, Python, or Piper subprocess synthesis** to the Java inference runtime.

This first implementation is an **import / phoneme / convolution reference
milestone**, not a runnable text-to-speech engine. No user-facing API should
claim successful synthesis before the complete VITS graph has native execution
and real-voice audio parity tests.

## Verified real Piper model

- Voice: `rhasspy/piper-voices`, `de/de_DE/thorsten/medium`
- Files: `de_DE-thorsten-medium.onnx` (~63 MB) and
  `de_DE-thorsten-medium.onnx.json`
- Language: German, eSpeak voice `de`
- Input contract: `input` (INT64 token IDs), `input_lengths` (INT64),
  `scales` (FLOAT32; noise scale, inverse speed/length scale, duration noise)
- Output: `output` (audio)
- Sampling rate: 22,050 Hz, one speaker
- Parsed with existing
  `com.aresstack.windirectml.windows.OnnxModelReader`:
  **2,682 nodes**, including 129 `Conv` and 3 `ConvTranspose`.
  The model was imported locally with `PiperOnnxModelSourceTest`.
- Model weights are **not committed**; `.gitignore` covers `model/piper-*/`.

Selected observed operators include `Conv`, `ConvTranspose`, `MatMul`,
`LeakyRelu`, `Sigmoid`, `Erf`, `Softmax`, `Gather`, `GatherND`,
`ScatterND`, `CumSum`, `Range`, `Pad`, `Expand`, `Slice`,
`RandomNormalLike`, `Where`, `Reshape`, `Transpose`, `Concat`,
`ReduceMean`, `Sqrt`, `Pow`, `Exp`, `Tanh` and `Softplus`.
These are **model requirements**, not a claim that every operation is already
implemented in our native execution engine.

## Implemented boundary

- `PiperVoiceConfig`: read paired Piper JSON voice configuration and
  validate eSpeak, sample rate, speakers, inference scalars and phoneme ID map.
- `PiperPhonemeIdEncoder`: turn IPA codepoints into Piper's beginning,
  separator and end IDs; reject unknown phonemes.
- `PiperSynthesisInput`: provide typed VITS input tensors and
  `noise_scale / length_scale / noise_w` in model order.
- `PiperPhonemizer`: phonemization port. A German eSpeak-NG adapter is still
  needed; the phoneme encoder does **not** implement text-to-phoneme rules.
- `PiperOnnxModelSource`: reuse the existing dependency-free ONNX protobuf
  importer, validate VITS inputs and outputs, retain node/tensor metadata and
  report the real model's required operator inventory.
- `PiperConv1dReference` and `PiperConvTranspose1dReference`: deterministic
  CPU baseline for grouped 1-D convolution and decoder upsampling, with
  synthetic output parity tests. These kernels are **not yet connected to a
  VITS graph executor or a DirectML execution path**.

The source works with the current Java 21 inference module. No ONNX Runtime
dependency is added.

## Next native VITS implementation steps

1. **Phonemizer.** Implement an eSpeak-NG phonemization adapter and verify
   IPA codepoint sequences against Piper's own voice-specific expectations.
   Keep the native phonemizer behind `PiperPhonemizer`.
2. **Compiled model.** Extend the existing ONNX-to-`.wdmlpack` architecture
   with Piper-specific graph operator metadata and tensor layouts; validate
   dynamic input/output shapes and all Constant node attributes. Do not load
   the ONNX model in the production execution path after compilation.
3. **Native operators.** Add DirectML kernels for 1-D/grouped Conv and
   ConvTranspose, plus required elementwise, reshape/broadcast, reductions,
   attention, indexing/scatter and random/sampling operations. Use the CPU
   reference implementations for numerical parity, not silent fallback.
4. **Real synthesis test.** Feed known German text with a recorded
   eSpeak phoneme-ID sequence into the whole graph, export the generated
   22,050 Hz waveform and compare its shape, length and audio output with a
   pinned Piper reference. Only then mark the voice `experimental` and expose
   a `PiperTextToSpeech` facade in `directml-inference`.
5. **Workbench / release.** Add voice selection, synthesized PCM playback and
   model-license provenance. Do not ship the example voice without checking
   its individual voice/dataset license.

## Tests

Run pure unit tests without a voice:

```powershell
.\gradlew.bat :directml-inference:test --tests 'com.aresstack.windirectml.inference.piper.*' --max-workers=1
```

For the gated real-model import test, place the model in
`model/piper-de_DE-thorsten-medium/` and run:

```powershell
cmd.exe /c ".\gradlew.bat :directml-inference:test --tests com.aresstack.windirectml.inference.piper.PiperOnnxModelSourceTest -Dpiper.realModel=true --max-workers=1"
```

The voice can be obtained from
https://huggingface.co/rhasspy/piper-voices/tree/main/de/de_DE/thorsten/medium.
