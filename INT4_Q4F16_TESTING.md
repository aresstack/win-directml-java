# Qwen INT4/q4f16 test notes

This source package switches the Qwen downloader to the Hugging Face q4f16 single-file ONNX artifact:

```text
onnx/model_q4f16.onnx -> model.onnx
```

The stale dense sidecar files are removed by the downloader:

```text
model.onnx_data
model.onnx.data
```

Expected runtime marker:

```text
Model format: INT4 quantized (MatMulNBits)
```

Useful JVM flags for the DirectML/Hybrid path:

```text
-Dqwen.gpu.layers=24
-Dqwen.gpu.lmhead=true
-Dqwen.gpu.attention=true
-Ddirectml.int4.gpu.disabled=false
```

Compare against the dequantized fallback with:

```text
-Ddirectml.int4.gpu.disabled=true
```

Notes:

* Run the Workbench Download tab with Force re-download once to replace an existing dense model file.
* Verify the log contains `INT4 quantized (MatMulNBits)`.
* Verify the log does not contain dense `Uploaded weight ... MB` lines for all large Qwen projection matrices.
