# WDMLPACK v1 Runtime Package

`*.wdmlpack` is the internal runtime package boundary for the Windows DirectML/WARP runtime.

The current v23 implementation supports two compatible package modes:

```text
mode=manifest-only, payloadIncluded=false
  v22 compatibility mode
  package is the runtime front door
  tensor payload still delegates to the source ONNX

mode=payload, payloadIncluded=true
  v23 native payload mode
  package contains tensor payload bytes
  runtime reconstructs the TensorCatalog and minimal Qwen graph from the package
  ONNX parsing is no longer needed on the package startup path
```

## Startup flow

```text
First start / no package:
  ONNX
  → QwenOnnxModelSource
  → TensorCatalog
  → QwenWdmlPackCompiler
  → model_q4f16.wdmlpack (header + JSON manifest + tensor payload)
  → normal v20/v22 load for this run

Next start / v23 package exists:
  model_q4f16.wdmlpack
  → QwenWdmlPackModelSource
  → mmap package
  → reconstruct minimal Qwen graph + tensor catalog
  → load weights from package payload
  → no ONNX graph parse
```

Existing v22 `manifest-only` packages are still accepted. When such a package is loaded and auto-create is enabled, the
compiler upgrades it to a payload package for the next start.

## Why this exists

ONNX is treated as an import format, not as the long-term runtime format. The runtime should start from a flat package
with prevalidated model metadata, a tensor directory and backend-specific layout metadata. This keeps the long-term
boundary clean:

```text
ONNX / SafeTensors / other import format
→ ModelSource / TensorCatalog
→ QwenWdmlPackCompiler
→ .wdmlpack
→ Runtime load layer
```

## Binary header

The v1 file starts with a fixed 64-byte little-endian header:

```text
0x00  8 bytes   ASCII magic "WDMLPACK"
0x08  int32le   container version (=1)
0x0c  int32le   flags (bit 0 = manifest-only)
0x10  int64le   manifest offset
0x18  int64le   manifest length
0x20  int64le   tensor directory offset (0; currently stored inside JSON manifest)
0x28  int64le   tensor directory length (0; currently stored inside JSON manifest)
0x30  int64le   payload offset (0 for manifest-only)
0x38  int64le   payload length (0 for manifest-only)
0x40  ...       UTF-8 JSON manifest
...   padding   zero padding to 4096-byte payload alignment
...   payload   concatenated tensor payloads
```

Tensor `payloadOffset` values in the JSON manifest are relative to the payload base from the header.

## Runtime properties

Runtime package loading is enabled by default when a matching package exists:

```text
-Dwindirectml.wdmlpack.load=true
```

Disable it with:

```text
-Dwindirectml.wdmlpack.load=false
```

Package auto-creation is enabled by default after a successful import:

```text
-Dwindirectml.wdmlpack.autoCreate=true
```

Disable auto-creation with:

```text
-Dwindirectml.wdmlpack.autoCreate=false
```

Payload packages are written by default in v23:

```text
-Dwindirectml.wdmlpack.payload=true
```

Force v22-style manifest-only packages with:

```text
-Dwindirectml.wdmlpack.payload=false
```

The explicit writer switch still exists:

```text
-Dwindirectml.wdmlpack.writeManifest=true
```

Optional custom output path:

```text
-Dwindirectml.wdmlpack.output=C:\path\to\model_q4f16.wdmlpack
```

Without a custom output path, the compiler writes next to the ONNX file:

```text
model_q4f16.onnx
model_q4f16.wdmlpack
```

## Manifest contents

The manifest stores:

```text
- package version and mode
- runtimeLoadable=true
- runtimeLoadMode
- source metadata for import provenance
- Qwen architecture metadata
- operator counts
- minimal runtime graph:
  - MatMulNBits nodes
  - Add nodes needed for connected bias recovery
- TensorCatalog summary
- sorted tensor directory:
  - name
  - ONNX dtype
  - dims
  - original storage kind
  - byte length
  - payloadOffset / payloadLength for v23 payload packages
  - source offset for external ONNX data when available
```

## Current scope

v23 stores the tensor payload in the package and removes ONNX graph parsing on the package startup path. It does not yet
prepack every weight into final backend-specific D3D12 upload layout. That is a later optimization layer on top of the
same package boundary.
