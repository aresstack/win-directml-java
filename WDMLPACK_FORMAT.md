# WDMLPACK v1 Skeleton

`*.wdmlpack` is the planned internal runtime package format for the Windows DirectML/WARP runtime.

The current v21 implementation is intentionally **manifest-only**:

```text
ONNX
→ QwenOnnxModelSource
→ TensorCatalog
→ QwenWdmlPackCompiler
→ model_q4f16.wdmlpack  (header + JSON manifest only)
```

The proven ONNX runtime path is not replaced yet. The package writer is only a preparation step for v22, where tensor
payloads can be copied into the package and the runtime can load the package directly.

## Why this exists

ONNX is now treated as an import format, not as the long-term runtime format. The runtime should eventually start from a
flat package with prevalidated model metadata, a tensor directory and backend-specific layout metadata.

## Binary header

The v1 file starts with a fixed 64-byte little-endian header:

```text
0x00  8 bytes   ASCII magic "WDMLPACK"
0x08  int32le   container version (=1)
0x0c  int32le   flags (bit 0 = manifest-only)
0x10  int64le   manifest offset
0x18  int64le   manifest length
0x20  int64le   tensor directory offset (0 in v21)
0x28  int64le   tensor directory length (0 in v21)
0x30  int64le   payload offset (0 in v21)
0x38  int64le   payload length (0 in v21)
0x40  ...       UTF-8 JSON manifest
```

## Enabling the optional writer

The writer is disabled by default so normal Workbench runs do not create large or surprising artifacts.

Enable it with:

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

The v21 manifest stores:

```text
- package version and mode
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
  - payloadOffset = -1 in v21 manifest-only mode
```

## Planned v22

v22 should fill the package payload and let the runtime load from `.wdmlpack`:

```text
model_q4f16.wdmlpack exists?
  yes → mmap package and load DirectML/WARP resources
  no  → import ONNX, write wdmlpack, load package
```

At that point ONNX remains only an import/fallback format.
