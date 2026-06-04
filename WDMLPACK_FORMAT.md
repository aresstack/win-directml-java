# WDMLPACK v1 Runtime Package

`*.wdmlpack` is the internal runtime package boundary for the Windows DirectML/WARP runtime.

The current v24 implementation supports two compatible package modes and adds cache hardening:

```text
mode=manifest-only, payloadIncluded=false
  v22 compatibility mode
  package is the runtime front door
  tensor payload still delegates to the source ONNX

mode=payload, payloadIncluded=true
  v23/v24 native payload mode
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

Next start / v24 package exists:
  model_q4f16.wdmlpack
  → QwenWdmlPackModelSource
  → mmap package
  → reconstruct minimal Qwen graph + tensor catalog
  → load weights from package payload
  → no ONNX graph parse
```

Existing v22/v23 packages without the v24 cache contract are treated as stale. They are deleted and rebuilt from ONNX
when the source ONNX is available. Manifest-only packages with a valid v24 cache contract can still act as a
compatibility front door and can be upgraded to a payload package.

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

Payload packages are written by default:

```text
-Dwindirectml.wdmlpack.payload=true
```

Force manifest-only packages with:

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
- v24 cache contract:
  - schema=qwen-wdmlpack-cache-v24
  - compilerVersion=24
  - source fingerprint based on file name, size, modified time and file key when available
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
  - payloadOffset / payloadLength for payload packages
  - source offset for external ONNX data when available
```

## Cache hardening

v24 rejects stale or incompatible packages before they can influence runtime loading. A package is considered stale when
the cache contract is missing, the compiler version does not match, or the source fingerprint no longer matches an
available ONNX source. Invalid packages are deleted and the loader falls back to ONNX so a fresh package can be
generated.

Writes are atomic: the compiler writes a unique temporary file, forces it to disk, then renames it into place. Directory
fsync is attempted best-effort where the platform permits it.

## Current scope

v24 stores tensor payload in the package, removes ONNX graph parsing on the package startup path, and hardens cache
invalidation/recovery. It does not yet prepack every weight into final backend-specific D3D12 upload layout. That is a
later optimization layer on top of the same package boundary.

## v25 SafeTensors import-only packages

v25 introduces SafeTensors as an import format, not as a direct runtime format.
A package compiled from raw SafeTensors metadata/payload must declare:

```json
"source": {"format": "safetensors"},
"runtimeLoadable": false,
"runtimeLoadMode": "import-only-tensor-catalog"
```

This prevents the Qwen runtime from accidentally treating a raw HF SafeTensors
layout as if it were the internal DirectML/WARP layout. The intended later flow
is:

```text
SafeTensors + config.json
-> TensorCatalog
-> Qwen WeightPacker / layout compiler
-> runtime-loadable .wdmlpack
```
