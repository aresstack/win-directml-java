# SafeTensors Import Skeleton (v25)

v25 adds the first SafeTensors import boundary. This is intentionally **not** a
runtime switch yet. The production Qwen path remains:

```text
wdmlpack payload cache -> ONNX fallback when stale/missing
```

The new pieces are:

```text
SafeTensorsReader
  - parses the SafeTensors 8-byte little-endian header length
  - parses the JSON tensor directory
  - validates dtype / shape / data_offsets
  - exposes mmap-backed tensor payload slices

QwenSafeTensorsModelSource
  - discovers/loads .safetensors files
  - validates the Qwen embedding tensor shape against config.json-derived config
  - exposes tensors via QwenModelImport + TensorCatalog

QwenWdmlPackCompiler
  - can build an import-only TensorCatalog manifest for SafeTensors sources
  - marks raw SafeTensors packages as runtimeLoadable=false
```

## Why import-only?

Raw Hugging Face SafeTensors contain tensors, not the DirectML/WARP runtime
layout. They do not contain ONNX `MatMulNBits` nodes, Qwen-specific fused
operator mapping, or our packed INT4 layout. Therefore v25 deliberately does
not let the runtime execute directly from a raw SafeTensors package.

The intended path is:

```text
SafeTensors + config.json
-> TensorCatalog
-> Qwen layout compiler / WeightPacker
-> runtime-loadable .wdmlpack
-> Runtime
```

v25 implements the first two steps. A later step will add the actual Qwen layout
compiler that converts SafeTensors tensor names/layouts into the exact payload
layout expected by the existing kernels.

## Cache compatibility

v25 does not intentionally invalidate existing v24 runtime `.wdmlpack` caches.
The stable ONNX/wdmlpack Qwen path remains unchanged.
