# Plan: SmolLM2 first-use package build — correctness review & version gating

> Status: **Vorschlag / noch nicht umgesetzt.** Dieses Dokument hält die
> Analyse und den geplanten Fix fest, falls wir das später umsetzen.

## Context
The DirectMlWorkbench compiles a SmolLM2 model into `model.wdmlpack` the first
time the user runs inference (lazy compile from the downloaded SafeTensors).
The question: is that on-first-use package built correctly?

## What was verified (paths)
- `directml-workbench/.../runtime/SmolLM2WorkbenchRuntimeRunner.java`
  - `ensureExecutablePackage()` (L184–202): rebuild only when
    `!isExecutablePackage(packagePath)`; otherwise reuse existing file.
  - `isExecutablePackage()` (L204–213): true iff `open()` succeeds AND
    `executable()==true`; **all exceptions silently swallowed**.
- `directml-inference/.../smollm2/SmolLM2WdmlPackCompiler.java`
  `compile()` (L39–57): read config → validate supported → read tensors →
  validate layout → `writeWithDensePayload(...)`. Build itself is sound.
- `SmolLM2WdmlPackManifest.java`: writes `compilerVersion = 41`,
  `schemaVersion`, `runtimeLoadable`, dense `tensors[]` with offsets.
- `SmolLM2RuntimePackage.from()` (L36–41): `executable = manifest.runtimeLoadable`.
  `validate()` (L82–92) only checks `modelFamily`/`architecture`/`model` keys.
- `RuntimeModelPackage`: has `validateSourceFingerprint()` / `validateSourceSize()`
  but they are **never called** on the workbench path, and the SmolLM2 manifest
  never writes a `source` block, so they cannot run for SmolLM2 anyway.

## Verdict
The package *content* is built correctly: config + validated layout + dense
payload with correct per-tensor offsets. The compile path is fine.

The **reuse decision is the real gap**:
1. **No version gating.** An existing `model.wdmlpack` is reused whenever its
   stored `runtimeLoadable` flag is true. Neither `compilerVersion (41)` nor
   `schemaVersion` is compared against the current code constants. After a
   workbench upgrade that bumps `COMPILER_VERSION` or changes the payload
   layout (recent history shows heavy WARP/format churn), a **stale package is
   silently reused** instead of rebuilt → possibly wrong output, no warning.
2. **Silent swallow.** `isExecutablePackage` catches every exception and maps a
   corrupt/incompatible package to the same "missing" state — no diagnostic.
3. **No source integrity check.** If a SafeTensors file changes/corrupts, the
   old package is reused with no fingerprint check.

## Proposed fix
In `SmolLM2RuntimeRunner.isExecutablePackage` / `ensureExecutablePackage`,
treat a version mismatch as "not executable so rebuild":
- Read `compilerVersion` + `schemaVersion` from the opened package manifest and
  compare to `SmolLM2WdmlPackManifest.COMPILER_VERSION` and `SCHEMA_VERSION`.
- If the file exists but version differs (or open fails), and a compile source
  is present, rebuild (compile already runs with `force=true`).
- Log (not swallow) the reason the existing package was rejected.

Optional hardening: write a `source` fingerprint block in the SmolLM2 manifest
and call `validateSourceSize`/`validateSourceFingerprint` on reuse.

## Verification
- Unit test in `directml-inference` / workbench runner test: build a package,
  hand-edit manifest `compilerVersion` to an old value, assert
  `ensureExecutablePackage` rebuilds rather than reuses.
- Manual: run workbench summarizer once (builds pack), bump COMPILER_VERSION,
  rerun → confirm rebuild + correct output.
