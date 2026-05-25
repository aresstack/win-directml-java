# Maven artifact structure

This project uses Maven Central as the primary distribution channel for library and sidecar artifacts. GitHub Release zips remain convenience packages for users who want a ready-to-run sidecar or workbench distribution, but they are not the only way to obtain the Java 21 sidecar.

## Published Maven Central artifacts

All artifacts use group id `com.aresstack`.

| Artifact | Java | Published | Purpose |
|---|---:|---:|---|
| `directml-config` | 8-compatible API surface where applicable | yes | Shared configuration, limits and model registry types. |
| `directml-windows-bindings` | 21 preview | yes | Java FFM bindings for DXGI, D3D12, DirectML and low-level Windows runtime access. |
| `directml-inference` | 21 preview | yes | Experimental Phi-3 inference/summarizer path used by the sidecar. |
| `directml-encoder` | 21 preview | yes | MiniLM, WordPiece E5 and cross-encoder reranker implementations. |
| `directml-runtime` | 21 preview | yes | Public Java 21 in-process ML runtime facade for embeddings and reranking. |
| `directml-sidecar-protocol-java8` | 8 | yes | Shared Java-8-compatible protocol and validation types. |
| `directml-sidecar-client-java8` | 8 | yes | Java 8 process/RPC client for host applications. |
| `directml-sidecar` | 21 preview | yes | JSON-RPC sidecar adapter over `directml-runtime`, used by Java 8 hosts. |
| `directml-sidecar-workbench` | 8 | no | Swing diagnostics UI, distributed as a GitHub Release zip only. |

## Typical dependency choices

### Java 21 application, no sidecar

Use the direct runtime API:

```gradle
implementation 'com.aresstack:directml-runtime:<version>'
```

This mode calls embeddings and reranking in-process. It does not start JSON-RPC and does not require the Java 8 sidecar client.

### Java 8 host application

Use the Java 8 client plus a matching sidecar artifact or distribution:

```gradle
implementation 'com.aresstack:directml-sidecar-client-java8:<version>'
```

The host process must start a Java 21 sidecar, either from the Maven artifact `com.aresstack:directml-sidecar:<version>` resolved by the application/build tooling, or from a GitHub Release zip. The Java 8 host classpath must not contain Java 21 runtime modules.

### Sidecar process

The sidecar itself is a Java 21 application artifact:

```gradle
runtimeOnly 'com.aresstack:directml-sidecar:<version>'
```

It depends on `directml-runtime`, `directml-encoder`, `directml-windows-bindings`, `directml-inference` and `directml-sidecar-protocol-java8`.

## Compatibility rules

- Use matching versions for `directml-sidecar-client-java8`, `directml-sidecar-protocol-java8` and `directml-sidecar`.
- The Java 8 client remains Java 8 bytecode and must not depend on Java 21 runtime modules.
- The sidecar is the process boundary for Java 8 hosts.
- Model weights are never shipped in Maven Central artifacts.
- GitHub Release zips are convenience launch packages and should match the Maven artifact version.

## Non-published application

`directml-sidecar-workbench` remains GitHub-release-only for now. It is a diagnostics/demo UI, not a stable library API.
