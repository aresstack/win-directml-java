# Maven artifact structure

This project now publishes only the core inference artifacts as new Maven Central artifacts: the Java 21 runtime modules plus the shared `directml-config` module. The JSON-RPC sidecar, Java 8 bridge, protocol module and workbench remain in the repository for compatibility with the previous beta line, but they are no longer part of new Maven Central releases.

The intended downstream architecture is:

```text
Java 21 application / ACP sidecar / manager project
    -> directml-runtime
    -> directml-encoder
    -> directml-windows-bindings
    -> directml-config
    -> Windows CPU / DirectML inference
```

A future ACP sidecar or manager-agent project should consume this repository as a pure Java library. It should not depend on this repository's legacy JSON-RPC sidecar as its product boundary.

## Published Maven Central artifacts going forward

All artifacts use group id `com.aresstack`.

| Artifact | Java | Published going forward | Purpose |
|---|---:|---:|---|
| `directml-config` | 8-compatible API surface where applicable | yes | Shared configuration, limits and model registry types used by the core runtime. |
| `directml-windows-bindings` | 21 preview | yes | Java FFM bindings for DXGI, D3D12, DirectML and low-level Windows runtime access. |
| `directml-encoder` | 21 preview | yes | MiniLM, WordPiece E5 and cross-encoder reranker implementations. |
| `directml-runtime` | 21 preview | yes | Public Java 21 in-process ML runtime API for embeddings and reranking. |
| `directml-inference` | 21 preview | no | Legacy/experimental Phi-3 inference path from the beta line. Kept in source, not published in new releases. |
| `directml-sidecar-protocol-java8` | 8 | no | Legacy beta protocol/validation module. Kept in source, not published in new releases. |
| `directml-sidecar-client-java8` | 8 | no | Legacy Java 8 JSON-RPC client. Kept in source, not published in new releases. |
| `directml-sidecar` | 21 preview | no | Legacy JSON-RPC sidecar adapter. Kept in source, not published in new releases. |
| `directml-sidecar-workbench` | 8 | no | Legacy diagnostics UI. Kept in source, not published in new releases. |

## Dependency choices

### Java 21 application, ACP sidecar, or manager project

Use the direct runtime API:

```gradle
dependencies {
    implementation 'com.aresstack:directml-runtime:<version>'
}
```

This is the canonical path. It calls embeddings and reranking in-process and does not require JSON-RPC or a sidecar process.

### Legacy Java 8 bridge

The Java 8 bridge modules remain in the source tree to preserve the previous beta implementation, but they are not part of new Maven Central releases. A project that still needs that bridge should stay on the previous beta artifacts or build the legacy modules from source.

New integration work should prefer a separate Java 21 ACP/manager project that depends on `directml-runtime` directly.

## Compatibility rules

- New releases focus on the core Java API and Windows inference engine.
- The published core set contains Java 21 runtime modules plus the shared `directml-config` module.
- Sidecar/protocol/workbench modules are legacy beta code unless explicitly reactivated later.
- The Java 21 API must not require JSON-RPC, ACP, MCP, A2A or sidecar concepts.
- Model weights are never shipped in Maven Central artifacts.
- Existing source modules are kept to avoid deleting beta-era code, but they should not drive new product architecture.
