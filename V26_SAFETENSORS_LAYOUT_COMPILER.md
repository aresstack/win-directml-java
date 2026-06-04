# v26 SafeTensors Qwen layout compiler

v26 extends the v25 SafeTensors import skeleton with a Qwen-specific layout compiler.

The important separation remains unchanged:

```text
Foreign import format  -> ModelSource / TensorCatalog -> Qwen layout compiler -> wdmlpack manifest
Runtime hot path       -> wdmlpack / ONNX fallback as before
```

## What changed

- Added `QwenSafeTensorsLayoutCompiler`.
- SafeTensors manifests now include a `qwenLayout` section.
- The compiler maps Hugging Face Qwen2 tensor names to stable runtime roles:
    - `model.embed_tokens.weight`
    - `model.norm.weight`
    - per-layer RMSNorms
    - per-layer Q/K/V/O projections
    - per-layer gate/up/down MLP projections
    - optional Q/K/V biases
    - `lm_head.weight` or tied embedding fallback
- Complete dense F16/F32 layouts are marked as:

```text
runtimeLoadable=true
runtimeLoadMode=wdmlpack-native-dense-payload
```

- Incomplete layouts, shape mismatches, or BF16/currently unsupported dense dtypes remain safe:

```text
runtimeLoadable=false
runtimeLoadMode=safetensors-layout-only
```

## Why this is still conservative

This does not change the validated Qwen ONNX/wdmlpack runtime path. The normal workbench still starts from the v24
payload cache and falls back to ONNX only when the cache is stale or missing.

The SafeTensors path is now structurally prepared for runtime packages, but dense SafeTensors is not yet the desired
performance target. The long-term target remains a compiler step that converts imported tensors into the optimized
internal layout expected by WARP/DirectML kernels.

## Next likely step

v27 should either:

1. add an explicit SafeTensors-to-wdmlpack compile entry point/tool, or
2. start the real prepacking/quantization layer so SafeTensors packages become WARP-optimized rather than merely
   dense-runtime-loadable.
