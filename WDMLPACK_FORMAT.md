# WDMLPACK v1 Runtime Front Door

`*.wdmlpack` is the internal runtime package boundary for the Windows DirectML/WARP runtime.

The current v22 implementation is still **manifest-only**, but it is now runtime-loadable:

```text
First start / no package:
  ONNX
  → QwenOnnxModelSource
  → TensorCatalog
  → QwenWdmlPackCompiler
  → model_q4f16.wdmlpack  (header + JSON manifest)
  → normal v20 ONNX-backed payload load

Next start / package exists:
  model_q4f16.wdmlpack
  → QwenWdmlPackModelSource
  → validates Qwen config + source metadata
  → delegates tensor payload to source ONNX for v22
  → normal v20 mmap/heap-light payload path
```

This means ONNX is no longer the only runtime front door. The runtime can start from the package, validate the manifest,
and only then use the source ONNX as the tensor-payload backing store. A later package version can replace that final
payload delegate with native `.wdmlpack` tensor payloads without changing the model source boundary again.

## Why this exists

ONNX is treated as an import format, not as the long-term runtime format. The runtime should eventually start from a
flat package with prevalidated model metadata, a tensor directory and backend-specific layout metadata.

v22 deliberately avoids copying hundreds of MiB of tensor payload into the package. That keeps the proven v20
speed/heap-light path intact while making the package boundary real and testable.

## Binary header

The v1 file starts with a fixed 64-byte little-endian header:

```text
0x00  8 bytes   ASCII magic "WDMLPACK"
0x08  int32le   container version (=1)
0x0c  int32le   flags (bit 0 = manifest-only)
0x10  int64le   manifest offset
0x18  int64le   manifest length
0x20  int64le   tensor directory offset (0 in v22 manifest-only)
0x28  int64le   tensor directory length (0 in v22 manifest-only)
0x30  int64le   payload offset (0 in v22 manifest-only)
0x38  int64le   payload length (0 in v22 manifest-only)
0x40  ...       UTF-8 JSON manifest
```

## Runtime properties

Runtime package loading is enabled by default when a matching package exists:

```text
-Dwindirectml.wdmlpack.load=true
```

Disable it with:

```text
-Dwindirectml.wdmlpack.load=false
```

Package auto-creation is enabled by default after a successful ONNX import:

```text
-Dwindirectml.wdmlpack.autoCreate=true
```

Disable auto-creation with:

```text
-Dwindirectml.wdmlpack.autoCreate=false
```

The explicit writer switch still exists and forces a manifest write:

```text
-Dwindirectml.wdmlpack.writeManifest=true
```

Optional custom output path:

```text
-Dwindirectml.wdmlpack.output=C:\\path\\to\\model_q4f16.wdmlpack
```

Without a custom output path, the compiler writes next to the ONNX file:

```text
model_q4f16.onnx
model_q4f16.wdmlpack
```

## Manifest contents

The v22 manifest stores:

```text
- package version and mode
- runtimeLoadable=true
- runtimeLoadMode=wdmlpack-frontdoor-onnx-payload
- source format and source ONNX metadata
- Qwen architecture metadata
- operator counts, including MatMulNBits count
- TensorCatalog summary
- sorted tensor directory:
  - name
  - ONNX dtype
  - dims
  - storage kind
  - byte length
  - source offset for external ONNX data when available
  - payloadOffset = -1 in v22 manifest-only mode
```

## Next package version

The next major packaging step should add native tensor payloads:

```text
model_q4f16.wdmlpack exists?
  yes → mmap package and load DirectML/WARP resources without parsing ONNX tensors
  no  → import ONNX/SafeTensors, write wdmlpack payload, load package
```

At that point ONNX remains only an import/fallback format.
