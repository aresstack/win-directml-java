# AskAI Consumer Guide — directml-inference generation runtime (0.2.0)

How AskAI's Java-21 sidecar consumes the neutral local generation runtime. AskAI maps this neutral
contract onto its own `/api/chat` and `/api/generate`; it does not implement any SPI here.

## 1. Dependency

```gradle
implementation 'com.aresstack:directml-inference:0.2.0'
// transitively brings directml-model-catalog, directml-model-package, directml-config,
// directml-windows-bindings (all com.aresstack:*:0.2.0) + jackson-databind + slf4j-api.
```

Requires a Java 21 runtime with `--enable-preview --add-modules=jdk.incubator.vector` and (for GPU
backends) `--enable-native-access=ALL-UNNAMED` on Windows with a DirectML/D3D12 device.

## 2. Public API surface (the only supported package)

`com.aresstack.windirectml.inference.api`:
`GenerationRuntime`, `GenerationModelHandle`, `GenerationRequest`, `GenerationResult`,
`GenerationToken`, `GenerationTokenListener`, `GenerationFinishReason`, `LoadPolicy`,
`GenerationException`, `GenerationErrorCode`. Everything else in the module is implementation detail
with no compatibility guarantee. Backends use the neutral `com.aresstack.windirectml.catalog.CatalogBackend`.

## 3. Resolve the descriptor (catalog is the source of truth)

```java
import com.aresstack.windirectml.catalog.*;

LocalRuntimeModelDescriptor descriptor =
        LocalModelCatalog.findByRepositoryId("google-t5/t5-small"); // by HF repo id
// descriptor.runtimeFamily(), .supportedBackends(), .runtimePackageFileName(),
// .runtimeDirectoryName(), .status()==ModelStatus.RUNNABLE, .capabilities(), .chatTemplate()
```

Only offer `descriptor.isRunnable()` models. The model directory is where AskAI installed the model
(it must contain the compiled `*.wdmlpack` named `descriptor.runtimePackageFileName()` plus the
tokenizer/config assets).

## 4. Load a model (package-only, fail-closed)

```java
import com.aresstack.windirectml.inference.api.*;
import com.aresstack.windirectml.catalog.CatalogBackend;
import java.nio.file.Path;

try (GenerationModelHandle handle = GenerationRuntime.load(
        descriptor, Path.of(modelDir), CatalogBackend.CPU, LoadPolicy.PACKAGE_ONLY)) {
    ...
}
```

`LoadPolicy.PACKAGE_ONLY` (recommended) loads only from the `*.wdmlpack` and never falls back to raw
weights. `GenerationRuntime.load(descriptor, dir, backend)` is shorthand for PACKAGE_ONLY.

## 5. Send a prompt

```java
GenerationRequest request = GenerationRequest.builder("What is Java? Answer in one sentence.")
        .systemPrompt("You are a concise assistant.")  // optional; null for none
        .maxNewTokens(128)
        .build();
GenerationResult result = handle.generate(request);   // greedy / deterministic
String text = result.text();
GenerationFinishReason why = result.finishReason();   // STOP | LENGTH | CANCELLED | ERROR
int in = result.promptTokenCount(), out = result.generatedTokenCount();
CatalogBackend actual = result.backend();             // the backend actually used (AUTO may resolve)
```

The family chat template / task prefix is applied internally from the descriptor (ChatML for
Qwen/SmolLM2, Gemma turns for Gemma 3, Phi-3 roles, raw "summarize: …" for T5). For seq2seq (T5)
put the full task string in the user prompt.

## 6. Stream tokens

```java
handle.generate(request, token -> {
    emitToClient(token.text());      // token.text(), token.tokenId(), token.index()
});
// GenerationTokenListener.onComplete(GenerationResult) fires once at the end.
```

## 7. Close

`GenerationModelHandle` is `AutoCloseable` — use try-with-resources or call `close()` (idempotent) to
release native/runtime resources. Not thread-safe: one handle per thread, or serialize calls.

## 8. Backends per family (catalog matrix — 0.2.0)

| Family | Allowed backends |
|---|---|
| Qwen, T5/CodeT5 | WARP, AUTO, CPU |
| SmolLM2 | AUTO, CPU (software-WARP withheld, problems.md P1) |
| Gemma 3 | WARP, AUTO (no CPU — no Python bridge) |
| Phi-3 | CPU (DirectML/AUTO pending, problems.md P2) |

An explicit backend outside a family's matrix → `UNSUPPORTED_BACKEND`. Explicit CPU/WARP/DIRECTML are
honoured exactly (no silent switch); AUTO may resolve to CPU and reports the actual backend in the
result.

## 9. Error codes (`GenerationException.errorCode()`)

`INVALID_REQUEST`, `UNSUPPORTED_BACKEND`, `MODEL_DIRECTORY_NOT_FOUND`, `UNSUPPORTED_FAMILY`,
`PACKAGE_MISSING`, `PACKAGE_NOT_LOADABLE`, `RAW_WEIGHTS_FALLBACK_BLOCKED`, `MODEL_ASSETS_MISSING`,
`GATED_ACCESS_BLOCKED`, `INITIALIZATION_FAILED`, `GENERATION_FAILED`. Branch on `errorCode()`; do not
parse messages. Map these onto AskAI's transport error surface.
