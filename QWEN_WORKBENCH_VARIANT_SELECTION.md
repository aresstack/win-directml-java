# Qwen Workbench ONNX variant selection

The Workbench download tab now keeps Hugging Face ONNX filenames unchanged.

Examples:

- `model_q4f16.onnx`
- `model_q4.onnx`
- `model_int8.onnx`
- `model_fp16.onnx`
- `model.onnx` plus `model.onnx_data`

The active runtime file is stored in:

```text
qwen-selected-onnx.txt
```

under the Qwen model directory:

```text
%APPDATA%\.directml\model\qwen2.5-coder-0.5b-directml-int4
```

When the user changes the combo box in the Workbench, the selected filename is written to that file. The Qwen loader
reads the filename from that file and loads the selected ONNX file.

The Workbench shows the Hugging Face URL for the selected ONNX file. By default, only the selected ONNX file is
downloaded. Check "Download all ONNX files" to prefetch every known Hugging Face ONNX variant.

The PowerShell downloader supports the same layout:

```powershell
Use the Workbench Download tab to select `model_q4f16.onnx`. Use the all-variants option in the Workbench when all known Qwen ONNX files should be fetched.
```
