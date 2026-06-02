# Qwen Workbench ONNX model selector

This patch keeps the Hugging Face ONNX file names locally and makes the Workbench choose the Qwen ONNX file explicitly.

## Workbench behavior

- The Download tab now contains a Qwen2.5-Coder ONNX selector.
- The selected Hugging Face URL is shown directly in the UI.
- Downloading Qwen downloads only the selected ONNX file by default.
- The checkbox `alle ONNX-Varianten herunterladen` downloads all known ONNX variants only when the user opts in.
- The Summarizer tab uses the selected ONNX file through the shared Workbench model state.

## Default selection

The Workbench defaults to:

```text
model_q4f16.onnx
```

The selected file is saved locally with the same file name.

## PowerShell downloader

```powershell
.\scripts\download-qwen.ps1 -OnnxFile model_q4f16.onnx -Force -Validate
.\scripts\download-qwen.ps1 -OnnxFile model_int8.onnx -Force -Validate
.\scripts\download-qwen.ps1 -AllOnnxVariants
```

`model.onnx` is the only variant that also downloads `model.onnx_data`.

## Verification performed here

The Gradle wrapper could not be executed in this sandbox because it tried to download `gradle-9.0-bin.zip` from
`services.gradle.org` and network access is blocked.

I therefore ran a local `javac` smoke compile for the changed Workbench and Qwen classes with Java 21 preview flags and
dependency stubs for external libraries. That check passed.
