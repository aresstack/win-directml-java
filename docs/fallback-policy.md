# CPU / DirectML Fallback Policy

This document defines the contract between the sidecar and its clients
(workbench, Java-8 host applications) for choosing between the CPU and
DirectML execution paths.

CPU is **not** a test-only path: it is a fully supported local fallback.

## Modes

Each backend (`embed`, `rerank`) accepts the same three values via its
respective system property:

| Property           | Default | Accepted values           |
|--------------------|---------|---------------------------|
| `-Dembed.backend`  | `auto`  | `cpu`, `directml`, `auto` |
| `-Drerank.backend` | `auto`  | `cpu`, `directml`, `auto` |

Unknown values cause the sidecar to exit immediately with exit code `2`
(configuration error).

### `auto` (default)

1. Try to load the **DirectML** backend.
2. On any failure (missing `DirectML.dll`, no DML-capable adapter,
   load error) log a warning and fall back to the **CPU** backend.
3. If **both** DirectML and CPU fail to load, the sidecar exits hard
   with exit code `3`.

When the fallback path is taken, the `health` response reports the
fallback via two dedicated fields. Example for embeddings:

```json
{
  "embeddingBackend": "cpu",
  "embeddingReady": true,
  "embeddingFallback": true,
  "embeddingFallbackReason": "embed.backend=auto: DirectML unavailable, falling back to CPU – DirectML.dll not found"
}
```

The same shape applies to the reranker (`rerankerFallback`,
`rerankerFallbackReason`).

`lastError` is **not** populated for soft fallbacks – it stays
reserved for hard failures.

### `cpu` (forced)

- Load the CPU backend only.
- Never silently fall back to anything else.
- If the CPU backend fails to load (e.g. missing model directory),
  the sidecar exits with exit code `3` and `lastError` describes
  the reason. There is no retry on DirectML.

### `directml` (forced)

- Load the DirectML backend only.
- **Never** silently fall back to CPU – the whole point of forcing
  this mode is to surface DirectML problems instead of hiding them.
- If DirectML fails to load (missing DLL, no DML-capable adapter,
  driver issue), the sidecar exits with exit code `3` and
  `lastError` describes the reason.

## Health contract

The `health` method always reports the active backend and the
fallback signal for both subsystems:

| Field                     | Type    | Meaning                                                                |
|---------------------------|---------|------------------------------------------------------------------------|
| `embeddingBackend`        | string  | `"cpu"`, `"directml"`, `"none"`, `"error"`                             |
| `embeddingReady`          | boolean | encoder is loaded and can serve `embed` calls                          |
| `embeddingFallback`       | boolean | `true` when `auto` ended up on CPU after DirectML failed; else `false` |
| `embeddingFallbackReason` | string  | human-readable reason when `embeddingFallback=true`; absent otherwise  |
| `rerankerBackend`         | string  | same value set as for embeddings                                       |
| `rerankerReady`           | boolean | reranker is loaded                                                     |
| `rerankerFallback`        | boolean | same semantics as `embeddingFallback`                                  |
| `rerankerFallbackReason`  | string  | human-readable reason when `rerankerFallback=true`                     |
| `lastError`               | string  | last hard error from any subsystem; absent when there was none         |

Clients should treat `embeddingFallback` / `rerankerFallback` as
**informational** (display, telemetry) and `lastError` as
**actionable** (something is broken and needs attention).

The Java-8 client exposes the new fields on `HealthResult`:

```java
HealthResult h = client.health();
if (h.isEmbeddingFallback()) {
    log.warn("Sidecar fell back to CPU embeddings: {}",
             h.getEmbeddingFallbackReason());
}
```

The workbench `HealthPanel` shows `embeddingBackend`,
`embeddingFallback`, `rerankerBackend` and `rerankerFallback` directly
in its status grid so a user can see at a glance whether DirectML or
CPU is in use.

## CPU-only setup

CPU is the supported choice on machines without a DML-capable GPU,
without `DirectML.dll`, or on non-Windows test boxes.

To run the sidecar in CPU-only mode set both backend properties
explicitly:

```text
-Dembed.backend=cpu
-Drerank.backend=cpu
```

Required model directories (the sidecar probes these in order; you
can also point at a specific directory with the corresponding
`-D…modelDir` property):

| Family         | System property     | Default location                              |
|----------------|---------------------|-----------------------------------------------|
| MiniLM (embed) | `-Dminilm.modelDir` | `model/all-MiniLM-L6-v2/`                     |
| E5 (embed)     | `-De5.modelDir`     | `model/<variant>/` (see `-De5.model`)         |
| Reranker       | `-Drerank.modelDir` | `model/cross-encoder-ms-marco-MiniLM-L-6-v2/` |

Optional: skip the reranker entirely by not providing
`-Drerank.modelDir` and leaving `-Drerank.backend` unset (default
`auto`); in that case the sidecar starts without a reranker and the
`rerank` method reports `not-implemented`.

A typical CPU-only invocation looks like:

```bash
java --enable-preview --enable-native-access=ALL-UNNAMED \
     -Dembed.backend=cpu \
     -Drerank.backend=cpu \
     -jar directml-sidecar-all.jar
```

This works on any OS that can run Java 21 – DirectML is not loaded
at all in this configuration.

## Summary for clients

- Read `mode` / `embeddingBackend` / `rerankerBackend` to know which
  device is actually being used.
- Read `embeddingFallback` / `rerankerFallback` to know whether the
  active backend matches the user's preference.
- Treat `lastError` as the canonical "something went wrong" signal.
- A non-zero sidecar exit code (`2` config error, `3` initialisation
  failure) means the user's forced mode could not be satisfied and
  the process intentionally did not silently degrade.
