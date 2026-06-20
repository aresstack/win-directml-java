# Qwen q4f16 INT4 loader fix

This ZIP replaces the previous broken patch and is based again on the original uploaded project tree.

## What changed

- the Workbench Download tab now downloads `onnx/model_q4f16.onnx` from `onnx-community/Qwen2.5-Coder-0.5B-Instruct`
  and stores it locally as `model.onnx`.
- The downloader no longer requires `model.onnx_data` for the default Qwen artifact.
- Stale dense sidecars (`model.onnx_data`, `model.onnx.data`) are removed when the q4f16 single-file artifact is
  selected.
- `QwenModelDownloadConfig.DEFAULT` now represents the q4f16 single-file artifact instead of the dense `model.onnx` +
  `model.onnx_data` pair.
- `QwenModelDirValidator` now accepts both layouts:
    - single-file `model.onnx`
    - `model.onnx` plus external sidecar when the ONNX graph references external tensors
- `Qwen2Weights` now loads quantized MatMulNBits tensor payloads from either:
    - ONNX external data references
    - inline ONNX initializers returned by `OnnxModelReader`

## Compile check performed here

A local `javac` compile check was run for the changed Qwen loader classes and the changed Workbench downloader classes.
Gradle itself could not be run in the sandbox because the Gradle wrapper attempted to download Gradle from the internet,
which is blocked in this environment.

## Test on Windows

Run:

```powershell
Use the Workbench Download tab with Force re-download and validation.
```

Then start your Workbench/DirectML flow and look for:

```text
Model format: INT4 quantized (MatMulNBits)
```

If you still see:

```text
Model format: dense FLOAT16/FLOAT
```

then the loaded `model.onnx` is not the intended q4f16 MatMulNBits artifact.
