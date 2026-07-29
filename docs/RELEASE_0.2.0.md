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

## Gate before a real Central publish

Actual `publishAllPublicationsToCentralPortal` (via the tag-driven CI, `release.ps1 0.2.0`) must wait
until:

1. every model intended as RUNNABLE has a real green proof (Phi-3 real run pending — no local
   weights; Gemma real run pending — gated HF token; see problems.md P4/P6),
2. `ORG_GRADLE_PROJECT_signingInMemoryKey` / `signingInMemoryKeyPassword` and
   `centralUsername` / `centralPassword` are present,
3. the version `0.2.0` is approved and tagged.

Signing is skipped automatically for `publishToMavenLocal`; it is required only for the Central
publish task on a non-SNAPSHOT version.
