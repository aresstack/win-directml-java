# Release 0.2.0 — local-engine artifacts (prepared, not yet published)

Coordinated Maven Central release surface for the AskAI local engine. **Prepared and tested via
`publishToMavenLocal`; not published to Central** (per instruction — publish only after approval +
credentials + all RUNNABLE models proven).

## Coordinated version

All AresStack modules move to `0.2.0` (group `com.aresstack`), a minor bump over `0.1.1` (additive:
new published `directml-model-catalog`, first-time published `directml-inference`, new public
generation API, package-only load policy, catalog-driven family adapters, new model families). No
existing public API is removed; `0.1.1` consumers keep working.

## Published modules (all `com.aresstack:<name>:0.2.0`, each with jar + sources + javadoc + POM)

| Module | Java | Role |
|---|---|---|
| directml-model-catalog | 8 | neutral model catalog + installed-model manifest (new to publish) |
| directml-model-package | 21 | neutral wdmlpack contracts + artifact lifecycle |
| directml-config | 8/21 | shared config types |
| directml-windows-bindings | 21 | FFM D3D12/DirectML bindings |
| directml-encoder | 21 | encoders + cross-encoder reranker |
| directml-runtime | 21 | public facade (embeddings, reranking) |
| directml-inference | 21 | **neutral generation runtime** — public API in `…inference.api` (new to publish) |

The `directml-inference` POM depends only on published `com.aresstack:*:0.2.0` modules plus Maven
Central libraries (jackson-databind 2.17.1, slf4j-api 2.0.13) — no SNAPSHOT and no mavenLocal-only
dependencies, so a foreign reproducible build resolves cleanly.

## Reproduce the local release

```
gradlew -Pversion=0.2.0 clean build
gradlew -Pversion=0.2.0 publishToMavenLocal
```

## RUNNABLE-model certification gate

Every catalog entry that is `RUNNABLE` must have **either** a real green public-API package-only proof
(`REAL_CERTIFIED`) **or** an explicitly documented, accepted release exception. The current RUNNABLE set
and its evidence:

| Model | Evidence | Gate |
|---|---|---|
| all-MiniLM / E5 (embeddings), ms-marco-MiniLM-L6 (rerank) | shipped + real-tested | ✅ certified |
| T5 / Flan-T5 / CodeT5 | REAL_CERTIFIED | ✅ certified |
| SmolLM2 135M / 360M | REAL_CERTIFIED (CPU + AUTO) | ✅ certified |
| Qwen/Qwen2.5-Coder-0.5B-Instruct | PROVEN_RUNTIME_REUSED | ⚠️ **accepted exception** (see below) |

**Accepted release exception — Qwen.** `Qwen/Qwen2.5-Coder-0.5B-Instruct` ships as `RUNNABLE` with
test-run state `PROVEN_RUNTIME_REUSED`, **not** `REAL_CERTIFIED`: the public-API package-only run was
skipped by user decision. It is accepted for release because the adapter reuses the exact shipping
`QwenInferenceEngine` path. This exception is explicit and bounded to Qwen; no other `RUNNABLE` entry
may ship without a real green proof.

**Not RUNNABLE (no release obligation).** Phi-3 (`CONTRACT_TESTED`, no local weights, P2/P6) and Gemma 3
(`EXTERNALLY_BLOCKED`, gated HF token, P4) are catalog status **UNVERIFIED** — absent from
`LocalModelCatalog.runnable()`, so they are neither offered as local recommendations nor a release
blocker. They keep full descriptors/manifests/backends and return to `RUNNABLE` only after a real green
public-API package-only run. The L12 reranker stays UNVERIFIED pending its real CPU + WARP package-only
smoke (P3 is code-fixed; see problems.md and LOCAL_ENGINE_CERTIFICATION.md).

## Gate before a real Central publish

Actual `publishAllPublicationsToCentralPortal` (via the tag-driven CI, `release.ps1 0.2.0`) must wait
until:

1. the RUNNABLE-model certification gate above holds (every RUNNABLE entry is REAL_CERTIFIED or the
   documented Qwen exception),
2. `ORG_GRADLE_PROJECT_signingInMemoryKey` / `signingInMemoryKeyPassword` and
   `centralUsername` / `centralPassword` are present,
3. the version `0.2.0` is approved and tagged.

Signing is skipped automatically for `publishToMavenLocal`; it is required only for the Central
publish task on a non-SNAPSHOT version.
