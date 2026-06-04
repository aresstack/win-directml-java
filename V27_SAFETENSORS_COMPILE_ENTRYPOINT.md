# v27 – Qwen SafeTensors → wdmlpack compile entry point

v27 adds an explicit compile-time entry point for Hugging Face Qwen2 SafeTensors
model directories. It does **not** change the normal workbench/runtime startup
path.

Stable runtime path remains:

```text
existing payload .wdmlpack
→ ONNX fallback only when cache is missing or stale
```

New explicit compile path:

```text
HF Qwen2 directory
  config.json
  *.safetensors
      ↓
QwenSafeTensorsModelSource
      ↓
QwenSafeTensorsLayoutCompiler
      ↓
QwenWdmlPackCompiler
      ↓
model.wdmlpack
```

## API

```java
Path modelDir = Path.of("C:/models/qwen2.5-coder-0.5b");
Path output = modelDir.resolve("model.wdmlpack");

QwenWdmlPackCompileTool.CompileResult result =
        QwenWdmlPackCompileTool.compileSafeTensorsDirectory(modelDir, output);
```

By default the API refuses to write import-only packages. This protects callers
from accidentally deploying an analysis package that the runtime must not load.

To write an analysis/import-only package anyway:

```java
var options = new QwenWdmlPackCompileTool.CompileOptions(
        modelDir,
        output,
        true,   // payload
        true    // allowImportOnly
);
QwenWdmlPackCompileTool.compileSafeTensorsDirectory(options);
```

## CLI-style entry point

The class also has a small `main` method:

```text
com.aresstack.windirectml.inference.qwen.QwenWdmlPackCompileTool
```

Arguments:

```text
--model-dir <dir>       Hugging Face Qwen2 directory containing config.json and *.safetensors
--output, -o <file>     Target .wdmlpack file; defaults to <model-dir>/model.wdmlpack
--manifest-only         Write only an analysis manifest; requires --allow-import-only
--allow-import-only     Allow writing a package that the runtime must not load yet
```

Example:

```bash
java -cp "..." com.aresstack.windirectml.inference.qwen.QwenWdmlPackCompileTool \
  --model-dir C:\\models\\qwen2.5-coder-0.5b \
  --output C:\\models\\qwen2.5-coder-0.5b\\model.wdmlpack
```

## Boundary

v27 is still a compile/import feature. It intentionally does not auto-switch the
workbench to SafeTensors. Runtime startup remains controlled by the existing
wdmlpack cache flow and ONNX fallback logic.
