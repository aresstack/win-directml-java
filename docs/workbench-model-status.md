# Workbench model runtime status (WORKBENCH-MODEL-STATUS-1)

Audit of every generation model family in the Workbench: is it selectable, is it executable, which backend
path runs for WARP / AUTO / CPU, and is any stale `planned` / `probe` / `experimental` /
"Runtime integration is planned" wording or silent Python fallback left over. Gemma is the reference
(WARP/AUTO native, CPU legacy external, no visible research mode).

Audit-only slice: no runtime optimization, no migration. The state below is what the code does today; the
prior Gemma slices (PRODUCT-1/2 + closeout) and the homogeneous-lifecycle work already removed the
user-visible residue, so this slice confirms it and locks it in with `WorkbenchModelStatusAuditTest`.

## Status table

| Model family | Dropdown | WARP | AUTO | CPU | Registry status | Executable | Remark |
|--------------|----------|------|------|-----|-----------------|------------|--------|
| **Gemma 3 270M-it** | yes | native DirectML (WARP software adapter) | native DirectML (first hardware adapter, optional) | legacy external Python | EXPERIMENTAL (internal, not shown) | **yes — product** | Closed out. Download tab = `Gemma3PackageLifecycle` (Convert/READY). No visible research label (runtime label `native-warp`). |
| Gemma 3 270M (base) | yes | native (no product package) | native | legacy external Python | PLANNED | no | Base checkpoint; selecting it without a package → clear Download→Convert message (no silent Python). |
| **Qwen2.5-Coder-0.5B-it** | yes | native DirectML INT4 (`model_q4f16.wdmlpack`) | native DirectML INT4 | engine backend | EXPERIMENTAL | **yes** | Runnable by status (WORKBENCH-MODEL-STATUS-2) — the PLANNED-guard `qwenTestModel` exemption was removed; `isQwenTestModel` now only routes to `QwenInferenceEngine` (like the other families' routing). No Python. |
| Qwen2.5-Coder 1.5B / 3B | yes | — | — | — | PLANNED | no | "selectable but not executable yet" (honest guard message). |
| **SmolLM2-135M / 360M** | yes | native DirectML/WARP (dense projections on the D3D12 software rasterizer), CPU reference-runtime fallback when no WARP device | hardware GPU when present, else CPU reference runtime | reference runtime | EXPERIMENTAL | **yes — from `.wdmlpack`** | No Python. Missing package → clear "Use Download tab → Convert". Audited honest in SMOLLM2-PRODUCT-AUDIT-1 (see below). |
| **T5 / Flan-T5 / CodeT5** | yes | mixed: dense projections + LM-head on DirectML/WARP, rest on CPU reference | same on hardware adapter | validated Java reference seq2seq | EXPERIMENTAL | **yes — from `.wdmlpack`** | Seq2seq. No Python. `T5InferenceEngine`. CPU=certified path. WARP/AUTO mixed path: **all four curated models (t5-small, flan-t5-small, codet5-small, codet5-base-multi-sum) real-certified (CPU == WARP, greedy; T5-REALMODEL-CERT-1..4)**. Audited honest in T5-PRODUCT-AUDIT-1 (see below). |
| Phi-3 mini (onnx) | yes | blocked (gate) | blocked (gate) | blocked (gate) | PLANNED | **no — not executable in Workbench** | Selectable + downloadable, but the homogeneous artifact gate (`CompilerMissingLifecycle`) has no wdmlpack compiler for Phi-3, so `requireExecutablePackage(PHI3)` always fails. PLANNED so the Summarizer guard blocks it upfront. A native Java/DirectML decoder (`Phi3InferenceEngine`, no Python/ONNX Runtime) exists in the library/tests only. Corrected in PHI3-PRODUCT-AUDIT-1 (see below). |
| Phi-3.5 mini (onnx) | yes | — | — | — | PLANNED | no | "selectable but not executable yet" (no wdmlpack compiler). |

## Findings

- **No user-visible stale wording.** The dropdown lists model ids only (no status tooltip). The only visible
  research term had been the Gemma runtime label `native-warp-experimental` → already fixed to `native-warp`
  (PRODUCT-2) and the Gemma "probe" notes → "legacy external Python" (closeout). No `Runtime integration is
  planned` string exists anywhere in the repo (verified); locked by `WorkbenchModelStatusAuditTest`.
- **Executable models are not blocked.** The Summarizer's PLANNED guard only fires for genuinely
  not-executable PLANNED models (Phi-3.5, Qwen 1.5B/3B, Gemma base); Gemma/SmolLM2/T5/Phi-3 and Qwen 0.5B
  are EXPERIMENTAL (runnable). Qwen 0.5B is runnable by status — the `qwenTestModel` PLANNED-guard exemption
  was removed in WORKBENCH-MODEL-STATUS-2 (the predicate now only routes to `QwenInferenceEngine`). Locked by
  the audit test.
- **Not-executable models are clearly marked**, both in the Summarizer ("selectable but not executable yet.
  Status: planned…") and in the Download tab (`package compiler not implemented (downloadable, not
  executable)` for families without a compiler).
- **Python only where allowed.** The single Python path is the Gemma `Backend = CPU` legacy external
  runner. WARP/AUTO never use Python/Transformers/ONNX-Runtime; no silent Python fallback (a missing Gemma
  package fails with a clear Download→Convert message).
- **Download-tab status is correct for Gemma.** `DefaultModelArtifactService` wires `Gemma3PackageLifecycle`
  (`hasCompiler()=true` → Convert/READY). The old download/probe-only remnants — `Gemma3DownloadLifecycle`,
  `ModelArtifactRow.gemmaDownloadOnlyStatusText`, and the `downloadOnlyCandidate` "Download only" label — were
  **removed in WORKBENCH-MODEL-STATUS-3**. `NOT_SUPPORTED` now uniformly renders "Compiler missing" (package
  compiler not implemented) for any compiler-less family; Gemma is not one of them.

## SmolLM2 audit (SMOLLM2-PRODUCT-AUDIT-1)

SmolLM2 135M/360M are honest, executable Workbench paths — no Runtime-optimization changes, audit/labels only:
- **Selectable + runnable.** Both are `EXPERIMENTAL` → in the dropdown and runnable by status. The redundant
  `!smolLm2Model` PLANNED-guard exemption was removed (`SummarizerPanel`) — SmolLM2 is runnable by status, not
  by a hard-coded exception (consistent with the Qwen 0.5B fix).
- **Routing (honest, no overclaim).** `Backend = WARP` → native DirectML/WARP runtime (dense projections on
  the D3D12 software rasterizer; norms/RoPE/attention/KV stay on CPU), with a transparent CPU
  reference-runtime fallback when the WARP path is not executable (the panel prints `Runtime mode` +
  `Runtime fallback` with the precise reason). `Backend = AUTO` → native on a hardware GPU when present, else
  the reference runtime. `Backend = CPU` → reference runtime. **No Python anywhere.**
- **Missing package** → `SmolLM2WorkbenchRuntimeRunner` fails fast with "Missing SmolLM2 runtime package. Use
  Download tab → Convert" (never compiles, never silently falls back).
- **Stale wording fixed.** The readiness line "…kernel execution is the next WARP implementation step"
  (implying WARP is unimplemented — it is not; `SmolLM2NativeWarpExecutor` is the default) was replaced with
  an honest "WARP path prepared but not executable here; using the CPU reference runtime (<reason>)". The
  legacy `SmolLM2DirectMlWarpExecutor` "probe" ("kernels not implemented yet") is a diagnostic opt-in
  (`-Dsmollm2.warp.executor=probe`), not the product path. Registry notes updated to describe the native
  WARP + reference-fallback path.

## T5 audit (T5-PRODUCT-AUDIT-1)

T5/Flan-T5/CodeT5 are honest, executable Workbench paths — audit/labels only, no T5 runtime change:
- **Selectable + runnable.** All four curated models (`Salesforce/codet5-small`,
  `Salesforce/codet5-base-multi-sum`, `google-t5/t5-small`, `google/flan-t5-small`) are `EXPERIMENTAL` → in
  the dropdown and runnable by status. They are not `PLANNED`, so the Summarizer's PLANNED guard never blocks
  them and no T5 guard exemption exists (unlike the old Qwen/SmolLM2 exemptions there was never one to remove).
- **Routing (honest, no overclaim).** `Backend = CPU` → the **validated** Java reference seq2seq runtime
  (`T5Runtime.load`). `Backend = WARP / AUTO` → a **mixed** path (`T5Runtime.loadWarp` = encoder + decoder +
  LM-head WARP boundaries): the dense projections (attention/feed-forward + LM-head matvecs) run through
  DirectML on the WARP software adapter (WARP) or first hardware adapter (AUTO) via the shared
  `WarpDenseProjection` / `MatMulNBitsKernel`, while layer norms, attention softmax, and relative-position bias
  stay on the CPU reference path. (At audit time this mixed path was *not yet* correctness-certified and was
  surfaced as experimental; it has since been real-certified end-to-end for all four curated T5 models — see the
  certification sections below.) The panel prints the precise stage routing as the execution mode (e.g.
  `reference` or `warp-encoder-boundary+warp-decoder-boundary+warp-lm-head`). **No Python on any T5 path** (the
  engine consumes only the `.wdmlpack`; it never runs ONNX/PyTorch/HF through a foreign runtime).
- **Missing package** → `T5InferenceEngine` fails fast (lifecycle `validateOrThrowBeforeInference`; the panel's
  `validateT5ModelFiles` adds a clear "Download or compile the selected T5 model first" message). Runtime never
  compiles.
- **Stale wording fixed.** Added an honest panel NOTE that WARP/AUTO is the mixed, uncertified path and CPU is
  the validated one (previously the panel only said "runs from a prebuilt .wdmlpack" and never disclosed the
  mixed routing — a user picking WARP could assume a fully native runtime). Corrected the stale `T5Runtime`
  class javadoc ("API shell for the future T5 WARP runtime" — the WARP boundary pipelines exist) and the
  `UNSUPPORTED_MESSAGE` constant ("…not implemented yet" → "…not certified yet"; it is no longer thrown and is
  retained only as a javadoc anchor for the package-level certification scope). Registry notes already used
  internal "Experimental …" wording with no `planned`/`probe`/`not implemented`/`runtime integration` residue.

### T5 mixed-path correctness certification (T5-CORRECTNESS-CERT-1)

Measured the WARP/AUTO mixed path against the CPU reference (reference = ground truth). Opt-in only:
- **Synthetic device cert** (`-Dt5.correctness.cert=true`, needs a D3D12/DirectML device — runs on this host): a tiny
  T5 `.wdmlpack` is driven through the **same** package on `T5Runtime.load` (reference) vs `T5Runtime.loadWarp`
  (WARP mixed), comparing greedy token ids and LM-head logits. Result on this host (RTX 5080, WARP software
  adapter): **token ids identical** (`[5,5,5,5,5,5]` reference == WARP, `firstDivergent=none`) for both the
  relu and gated-gelu configs, and **LM-head `maxAbsLogitDiff = 0.0`, top-1 identical**. → The WARP dense-projection
  arithmetic (the part that actually moves to DirectML) is **bit-exact** vs the reference for the FP32 weight path.
  Caveat: the tiny synthetic weights make greedy collapse to a constant token, so the token-id parity is a
  low-entropy signal; the meaningful number is the `0.0` logit delta.
- **Real-model cert** (`-Dt5.realModel=true` + a local model dir): drives `google-t5/t5-small`,
  `google/flan-t5-small`, `Salesforce/codet5-small`, `Salesforce/codet5-base-multi-sum` through the engine on
  `reference` vs `warp` and compares token preview + text. **Skipped — no T5 artifact is present** locally (no
  `config.json`/safetensors and no compiled `model_t5.wdmlpack` under `model/`); the test disables cleanly via
  `@EnabledIf`.

**Verdict (T5-CORRECTNESS-CERT-1): D (data not yet sufficient at that time).** The synthetic cert showed the WARP
boundary arithmetic is faithful (bit-exact FP32 dense projections), but no real-model package was present to measure
end-to-end text parity. This was resolved in T5-REALMODEL-CERT-1 (below).

### T5 real-model certification — entire curated family (T5-REALMODEL-CERT-1..4)

Downloaded `google-t5/t5-small` (public repo, the existing Download/Convert path) into `model/t5-small`, compiled
`model_t5.wdmlpack` (132 tensors, reference generation verified end-to-end), and ran the gated real-model cert:

```
./gradlew :directml-inference:test --tests '*T5MixedRuntimeCorrectnessCertTest' \
  -Dt5.correctness.cert=true -Dt5.realModel=true
```

Result on this host (RTX 5080, WARP software adapter), prompt `"translate English to German: The house is
wonderful."`, maxTokens 20, greedy (temperature 0), executionMode
`warp-encoder-boundary+warp-decoder-boundary+warp-lm-head`:

| | CPU reference | WARP mixed |
|---|---|---|
| token ids | `[644:▁Das, 4598:▁Haus, 229:▁ist, 19250:▁wunderbar, 5:.]` | identical |
| text | `Das Haus ist wunderbar.` | identical |
| first divergent token | — (none) | — |

`tokensMatch=true`, `textMatch=true`. Backed by the synthetic LM-head cert (`maxAbsLogitDiff=0.0`, top-1 identical).
The cert now **asserts** parity for any present model (it fails if WARP diverges), so this is a locked guard, not
just a report. AUTO was not measured separately (it uses the same `loadWarp` mixed logic on the hardware adapter;
WARP already certified the arithmetic).

**google/flan-t5-small (T5-REALMODEL-CERT-2)** — downloaded via the same path into `model/flan-t5-small`, compiled
`model_t5.wdmlpack` (190 tensors, reference-verified), same cert run. Prompt `"Answer the question: what is the
capital of France?"`, maxTokens 20, greedy, same executionMode:

| | CPU reference | WARP mixed |
|---|---|---|
| token ids | `[1410:▁France]` | identical |
| text | `France` | identical |
| first divergent token | — (none) | — |

`tokensMatch=true`, `textMatch=true`.

**Salesforce/codet5-small (T5-REALMODEL-CERT-3)** — the CodeT5 special case: weights from `pytorch_model.bin` (the
restricted Torch state-dict import path) and a BPE tokenizer (`vocab.json` + `merges.txt`), not safetensors +
SentencePiece. Downloaded into `model/codet5-small`, compiled `model_t5.wdmlpack` (134 tensors, reference-verified)
— the Torch/BPE artifacts were processed cleanly. Same cert run. Prompt `"def add(a, b):\n    return a + b"`,
maxTokens 20, greedy, same executionMode:

| | CPU reference | WARP mixed |
|---|---|---|
| token ids | `[1:<s>, 32099:<extra_id_0>, 30::]` | identical |
| text | (base-checkpoint sentinel output) | identical |
| first divergent token | — (none) | — |

`tokensMatch=true`, `textMatch=true`. Note: codet5-small is a *pretrained base* checkpoint (not fine-tuned for
summarization), so it emits span-corruption sentinels (`<s> <extra_id_0> :`) — low output quality but **identical on
CPU and WARP**, which is what the cert measures.

**Salesforce/codet5-base-multi-sum (T5-REALMODEL-CERT-4)** — the largest curated T5 (base, ~990 MB payload, 260
tensors; Torch checkpoint + BPE). Downloaded into `model/codet5-base-multi-sum`, compiled `model_t5.wdmlpack`
(reference-verified), same cert run (passed within the 2 GB test heap — models load and shut down sequentially).
Prompt `"def add(a, b):\n    return a + b"`, maxTokens 20, greedy, same executionMode:

| | CPU reference | WARP mixed |
|---|---|---|
| token ids | `[1:<s>, 3495:Sum, 21872:marize, 2795:Ġtwo, 6548:Ġterms, 263:Ġ.]` | identical |
| text | `Summarize two terms.` | identical |
| first divergent token | — (none) | — |

`tokensMatch=true`, `textMatch=true`. This model is fine-tuned for summarization, so the output is coherent (unlike
the base codet5-small) — and identical on both paths.

**Verdict: A — all four curated T5 models (google-t5/t5-small, google/flan-t5-small, Salesforce/codet5-small,
Salesforce/codet5-base-multi-sum) WARP mixed are end-to-end correctness-equal to the CPU reference.** With CERT-4 the
entire curated T5 family is real-certified; the panel NOTE now states all four are certified (no T5 model is left in
the experimental/uncertified group). The downloaded artifacts under `model/t5-small/`, `model/flan-t5-small/`,
`model/codet5-small/` and `model/codet5-base-multi-sum/` are git-ignored. Tests:
`T5MixedRuntimeCorrectnessCertTest` (`directml-inference`). See `t5-realmodel-cert.md` for the prep recipe.

### T5 product status (T5-PRODUCT-CLOSEOUT)

Final state of the curated T5/Flan-T5/CodeT5 family as a Workbench product path:
- **All four curated models are real-certified** (CPU reference == WARP mixed, greedy, token ids + text identical):
  `google-t5/t5-small`, `google/flan-t5-small`, `Salesforce/codet5-small`, `Salesforce/codet5-base-multi-sum`
  (T5-REALMODEL-CERT-1..4). The gated cert asserts this parity for every present model.
- **CPU** = the always-validated Java reference seq2seq runtime. **WARP/AUTO** = the mixed DirectML path (dense
  projections on the WARP software adapter / first hardware adapter, the rest on the CPU reference), now
  real-certified against CPU for the four curated models. **No Python** on any T5 path.
- **A prebuilt `.wdmlpack` is mandatory** (the engine never compiles at inference) and **Download → Convert is the
  prerequisite** to produce it. Missing package → clear fail-fast message.
- **Certification scope / product boundary:** the certification covers exactly these four curated models. Any other
  or newly-added T5/CodeT5 model is **not** automatically certified — it must run its own real-model cert
  (`T5MixedRuntimeCorrectnessCertTest`, opt-in) before being described as certified. No blanket guarantee for
  arbitrary T5-family models.
- **Gating / artifacts:** the real-model and synthetic certs stay opt-in (`-Dt5.realModel=true`,
  `-Dt5.correctness.cert=true`); the standard regression stays light. Downloaded `model/...` artifacts are
  git-ignored and never committed.

## Phi-3 audit (PHI3-PRODUCT-AUDIT-1)

The Phi-3 family was advertised as runnable but is **not executable in the Workbench** — corrected to honest status
(audit/labels/docs only, no runtime change):
- **Root cause.** `SummarizerPanel.runPhi3Summarizer` calls `WorkbenchArtifactGate.requireExecutablePackage(PHI3, …)`
  first. Phi-3's lifecycle is `CompilerMissingLifecycle` (no wdmlpack compiler): `inspect()` is always
  `PACKAGE_COMPILER_MISSING` / `executable=false`, so `ready()` is false and the gate **always throws** "Package
  compiler not implemented … downloadable but not executable until a wdmlpack compiler exists." The homogeneous
  lifecycle deliberately does **not** run families from raw weights, so Phi-3 has no executable Workbench path today.
- **False promise fixed.** Phi-3-mini was `EXPERIMENTAL` (runnable by status), so the Summarizer's PLANNED guard let
  it through and the user hit the gate's deep error only after selecting + downloading. Downgraded **Phi-3-mini →
  PLANNED** (Phi-3.5 was already PLANNED), so the guard now blocks both upfront with the honest "selectable but not
  executable yet" message. No status now claims a runnable Phi-3.
- **Misleading note fixed.** The registry note claimed "Uses ONNX Runtime GenAI for text generation" — wrong: there
  is no ONNX Runtime / GenAI and no Python. The real `Phi3InferenceEngine` is a native Java/DirectML decoder that
  memory-maps ONNX-format weights (`model.onnx` + `model.onnx.data`); it runs only via the library/tests, not the
  Workbench product path. Notes reworded for both Phi entries.
- **Routing.** All of WARP/AUTO/CPU are blocked by the gate in the Workbench (no executable path). The library engine
  supports `cpu` / `directml` / `auto` (projections on GPU, attention/norms on CPU), but that is not the product
  surface. **No Python on any Phi path.**
- **Status:** PLANNED for both (selectable + downloadable, not executable). Making Phi-3 executable needs a wdmlpack
  compiler (a separate runtime slice, out of scope here). Locked by
  `WorkbenchModelStatusAuditTest.phiModelsAreNotRunnableAndCarryHonestNotes` and the config
  `GenerationModelRegistryTest`.
- **Compiler requirements specced (PHI-WDMLPACK-COMPILER-AUDIT-1).** What a minimal `model_phi3.wdmlpack` compiler
  needs (decision **B**: real new compiler, runtime largely reusable; Qwen's ONNX(INT4)→wdmlpack is the template) is
  documented in `phi3-wdmlpack-compiler-plan.md`. The Workbench status stays PLANNED until that compiler exists.
- **Compiler foundation built + heap-safe (PHI3-WDMLPACK-COMPILER-1/2).** `Phi3WdmlPackCompiler` +
  `Phi3WdmlPackRoles` + `Phi3RuntimePackage` (+ `Phi3Weights.ofRecords`) + a streaming layout planner
  (`Phi3Weights.planLayout`/`Phi3WeightLayout`) exist; synthetic round-trip is byte-exact. COMPILER-2 made the real
  compile **stream from the ONNX mmap** so it runs in the standard 2 GB heap — the gated real Phi-3-mini compile
  produces a real ~2.39 GB `model_phi3.wdmlpack` (711 tensors, 32 layers). **Known follow-up:** reloading a >2 GB
  package is blocked by the shared `WdmlPackReader` 2 GB mmap limit (needs a shared-reader slice). **No Workbench
  wiring / no status flip** — Phi-3 stays PLANNED / gate-blocked. See `phi3-wdmlpack-compiler-plan.md`. Then
  `PHI3-WORKBENCH-RUNNABLE-1`.

## Closeout (WORKBENCH-MODEL-STATUS-CLOSEOUT-1)

Final state of the whole generation-model status strand. Audit/docs/lock-in only — no runtime, kernel, format,
downloader or UI change in this slice.

**Runnable by status (SHIPPED/EXPERIMENTAL — verified executable Workbench path):**
- `google/gemma-3-270m-it` — native Java/DirectML (WARP/AUTO); CPU = legacy external Python.
- `Qwen/Qwen2.5-Coder-0.5B-Instruct` — native DirectML INT4 (`model_q4f16.wdmlpack`); no Python.
- `HuggingFaceTB/SmolLM2-135M-Instruct`, `HuggingFaceTB/SmolLM2-360M-Instruct` — native DirectML/WARP dense
  projections + CPU reference-runtime fallback; no Python.
- `google-t5/t5-small`, `google/flan-t5-small`, `Salesforce/codet5-small`, `Salesforce/codet5-base-multi-sum` —
  mixed DirectML/WARP path, all four real-certified CPU == WARP (T5-REALMODEL-CERT-1..4); no Python.

**PLANNED / not executable (selectable + clearly blocked by the Summarizer guard):**
- `Qwen/Qwen2.5-Coder-1.5B-Instruct`, `Qwen/Qwen2.5-Coder-3B-Instruct` — no runtime/artifacts yet.
- `google/gemma-3-270m` (base checkpoint) — routes to the Gemma handler for a clear missing-package message.
- `microsoft/Phi-3-mini-4k-instruct-onnx`, `microsoft/Phi-3.5-mini-instruct-onnx` — no wdmlpack compiler; the
  artifact gate blocks raw-weight execution (the native engine runs only via the sidecar `summarize` path).

**Stale-wording sweep (no false product promises remain):** searched the tree for `Runtime integration is planned`,
`not executable yet`, `not implemented yet`, `probe`, `download only`, `uses ONNX Runtime`, false Python/native-WARP
claims, and wrong EXPERIMENTAL/runnable status. Remaining hits are all legitimate: the honest PLANNED guard message
("selectable but not executable yet"), historical chronicle entries, the opt-in SmolLM2 `probe` diagnostic's own
accurate message, and the negation "no Python/ONNX Runtime". No visible status note contradicts the actual runtime
path.

**Invariants locked by tests:** `WorkbenchModelStatusAuditTest` pins the exact runnable set
(`runnableSetIsExactlyTheKnownProductModels`), the no-stale-wording rule, the executable-vs-PLANNED split, the
SmolLM2/T5/Phi honest-note rules; the config `GenerationModelRegistryTest` pins the per-model statuses. Real-model
certs stay opt-in (`-Dt5.realModel`/`-Dt5.correctness.cert`); downloaded `model/...` artifacts are git-ignored.

## Gemma reference (unchanged)

Gemma stays the reference product path and is untouched by this slice: `Backend = WARP` → native
Java/DirectML on the WARP software adapter (no Python); `Backend = AUTO` → native on the first hardware
adapter (optional, clear message when none); `Backend = CPU` → legacy external Python only. Paris smoke
(" Paris", token 9079) green on WARP and HARDWARE (gated tests). See `gemma3-warp-runtime-plan.md`
(Status: COMPLETE).
