# Gemma 3 270M-it — native Java/WARP runtime: progress & open points

Running log for the GEMMA-WARP-* slices (see `directml-inference/docs/gemma-3-270M.md` for the plan).
Each slice = own commit. Blockers are recorded here and the next buildable step is taken instead of
stopping. Open points are resolved with the user at the end.

## Confirmed config (real, from `gemma-3-270m-it/config.json`)

| field | value |
|-------|-------|
| model_type / arch | `gemma3_text` / `Gemma3ForCausalLM` |
| hidden_size | 640 |
| intermediate_size | 2048 |
| num_hidden_layers | 18 |
| num_attention_heads | 4 |
| num_key_value_heads | 1 (GQA) |
| head_dim | 256 (decoupled: 4×256=1024 ≠ 640) |
| vocab_size | 262144 (256k) |
| max_position_embeddings | 32768 |
| sliding_window | 512 |
| _sliding_window_pattern | 6 (full attention at layers 5, 11, 17) |
| rope_theta (global) / rope_local_base_freq | 1_000_000 / 10_000 |
| query_pre_attn_scalar | 256 → attn scale 1/sqrt(256) |
| rms_norm_eps | 1e-6 (zero-centered RMSNorm) |
| hidden_activation | gelu_pytorch_tanh (GeGLU) |
| bos/eos/pad | 2 / 1 / 0 |
| softcapping | none (attn & final null) |
| tie_word_embeddings | (absent → Gemma ties embed↔lm_head) |

## Status: COMPLETE

The Gemma 3 270M-it WARP line is finished end to end:
- **Speed** (GEMMA-WARP-13*–19 + closeout): submits/fences/readbacks minimized, projection groups fused;
  WARP decode at its fp32 DML-GEMM compute ceiling (~500 ms/token). Quantization is not a WARP speed lever.
- **Host memory** (GEMMA-BF16-PACK-1–3 + closeout): retained host weights are BF16 end to end (~511 MB less
  system RAM), lossless, no `.wdmlpack` format change, decode unchanged.
- **Product path** (GEMMA-PRODUCT-1–2 + closeout): Gemma is a normal Workbench model —
  `Backend = WARP` runs the native Java/DirectML runtime on the explicit WARP software adapter (CPU-only, no
  Python/Transformers/ONNX); `Backend = AUTO` runs the same native runtime on the first hardware adapter
  (optional accelerator, clear message when none); `Backend = CPU` is the legacy external Python path only.
  Missing `model_gemma3.wdmlpack` → a clear Download→Convert message (no silent Python fallback). The gated
  HF token lives in `%APPDATA%/.directml/download-overrides.json` (`huggingFaceTokens`, sent only to
  huggingface.co; 401/403 → clear gated-access hint). No visible runtime selector, profiling toggle, research
  mode, or `experimental`/`probe`/`planned` wording. Paris smoke (" Paris", 9079) green on WARP and HARDWARE.

## Slice status

| Slice | Status |
|-------|--------|
| T5-REALMODEL-CERT-4 certify Salesforce/codet5-base-multi-sum CPU vs WARP (real) | **done (ran the real cert + honest note/doc update; no runtime/kernel change). Largest curated T5 (base, ~990MB payload, 260 tensors; Torch checkpoint + BPE). Downloaded codet5-base-multi-sum via the existing public path into model/codet5-base-multi-sum, compiled model_t5.wdmlpack (reference-verified), ran the same gated cert (passed within the 2GB test heap; models load+shutdown sequentially). Result (RTX 5080/WARP, prompt "def add(a, b): return a + b", greedy, mode warp-encoder-boundary+warp-decoder-boundary+warp-lm-head): CPU reference == WARP mixed EXACTLY — token ids [1:<s>,3495:Sum,21872:marize,2795:Ġtwo,6548:Ġterms,263:Ġ.], text "Summarize two terms." (coherent — fine-tuned model), tokensMatch=true textMatch=true, first divergent=none; synthetic LM-head maxAbsLogitDiff=0.0. **Verdict A: codet5-base-multi-sum WARP mixed is end-to-end correctness-equal.** With CERT-4 the ENTIRE curated T5 family (t5-small, flan-t5-small, codet5-small, codet5-base-multi-sum) is real-certified+asserted; no T5 model left experimental/uncertified. Panel NOTE now states all four certified. Reverted the one-off t5CompileTmp task. Other 3 T5 certs unchanged. Gemma/Qwen/SmolLM2/Phi unchanged. 4-module regression green; gated real+synthetic cert green on this host.** |
| T5-REALMODEL-CERT-3 certify Salesforce/codet5-small CPU vs WARP (real) | **done (ran the real cert + honest note/doc update; no runtime/kernel change). CodeT5 special case: downloaded codet5-small (pytorch_model.bin Torch checkpoint + vocab.json/merges.txt BPE, no safetensors/SentencePiece) into model/codet5-small via the existing public path, compiled model_t5.wdmlpack (134 tensors, reference-verified) — Torch/BPE artifacts processed cleanly. Cert (RTX 5080/WARP, prompt "def add(a, b): return a + b", greedy, mode warp-encoder-boundary+warp-decoder-boundary+warp-lm-head): CPU reference == WARP mixed EXACTLY — token ids [1:<s>,32099:<extra_id_0>,30::], tokensMatch=true textMatch=true, first divergent=none; synthetic LM-head maxAbsLogitDiff=0.0. (Output is base-checkpoint span-corruption sentinels = low quality but byte-identical on both paths; cert measures parity not quality.) **Verdict A: codet5-small WARP mixed is end-to-end correctness-equal.** t5-small+flan+codet5-small now certified+asserted; codet5-base-multi-sum still experimental/uncertified (NOT upgraded). Refined panel NOTE + status doc. Reverted the one-off t5CompileTmp task. t5-small/flan-t5-small status unchanged. Gemma/Qwen/SmolLM2/Phi unchanged. 4-module regression green; gated real+synthetic cert green on this host.** |
| T5-REALMODEL-CERT-2 certify google/flan-t5-small CPU vs WARP (real) | **done (ran the real cert + honest note/doc update; no runtime/kernel change). Downloaded google/flan-t5-small via the existing public path into model/flan-t5-small, compiled model_t5.wdmlpack (190 tensors, reference-verified), ran the same gated cert. Result (RTX 5080/WARP, prompt "Answer the question: what is the capital of France?", greedy, mode warp-encoder-boundary+warp-decoder-boundary+warp-lm-head): CPU reference == WARP mixed EXACTLY — token ids [1410:▁France], text "France", tokensMatch=true textMatch=true, first divergent=none; synthetic LM-head maxAbsLogitDiff=0.0. **Verdict A: flan-t5-small WARP mixed is end-to-end correctness-equal.** Both t5-small AND flan-t5-small now certified+asserted by the cert; CodeT5 still experimental/uncertified (NOT upgraded). Refined panel NOTE + status doc accordingly. Added model/flan-t5-small to the git-ignored set (already covered). Reverted the one-off t5CompileTmp build task. t5-small status unchanged (still certified). Gemma/Qwen/SmolLM2/Phi unchanged. 4-module regression green; gated real+synthetic cert green on this host.** |
| T5-REALMODEL-CERT-1 certify google-t5/t5-small CPU vs WARP (real) | **done (ran the real cert + honest note/doc update; no runtime/kernel change). Downloaded google-t5/t5-small via the existing public Download path into model/t5-small, compiled model_t5.wdmlpack (132 tensors, reference-verified), ran `:directml-inference:test --tests '*T5MixedRuntimeCorrectnessCertTest' -Dt5.correctness.cert=true -Dt5.realModel=true`. Result (RTX 5080/WARP, prompt "translate English to German: The house is wonderful.", greedy, mode warp-encoder-boundary+warp-decoder-boundary+warp-lm-head): CPU reference == WARP mixed EXACTLY — token ids [644:▁Das,4598:▁Haus,229:▁ist,19250:▁wunderbar,5:.], text "Das Haus ist wunderbar.", tokensMatch=true textMatch=true, first divergent=none; synthetic LM-head maxAbsLogitDiff=0.0. **Verdict A: google-t5/t5-small WARP mixed is end-to-end correctness-equal.** Cert now ASSERTS parity for any present model (locked guard). Refined panel NOTE + status doc: t5-small certified, Flan/CodeT5 still experimental/uncertified (NOT pauschal upgraded). Added model/t5-small + other T5 dirs to .gitignore (downloaded artifacts not committed). Reverted the one-off t5CompileTmp build task. Gemma/Qwen/SmolLM2/Phi unchanged. 4-module regression green; gated real+synthetic cert green on this host.** |
| T5-REALMODEL-CERT-PREP-1 prepare google-t5/t5-small for the real-model cert | **done (verification + test ergonomics + docs; no runtime/downloader/convert change). Verdict B: the full Download→Convert chain for `google-t5/t5-small` is already wired (DownloadPanel "Download google-t5/t5-small (SafeTensors smoke-test)" → manifestForGoogleT5Small from public hf.co/google-t5/t5-small/resolve/main; ModelArtifactRow(T5, …, T5PackageLifecycle) Convert → model_t5.wdmlpack in <modelRoot>/t5-small). Expected artifacts: model.safetensors+config.json (convert source), tokenizer.json/tokenizer_config.json/spiece.model (tokenizer), model_t5.wdmlpack (produced). Cert prep: T5MixedRuntimeCorrectnessCertTest auto-resolves <repo>/model/t5-small and now also honors a clean generic `-Dt5.testModelDir` for the primary model (plus per-model `t5.testModelDir.<id>`). The only reason CERT-1 returned D was that nothing had been downloaded — not a missing building block; performing the network download is all that remains. Docs: new docs/t5-realmodel-cert.md (recipe + exact invocation + artifact table) + this row. Flan-T5/CodeT5 analogous (skip cleanly until downloaded). Gemma/Qwen/SmolLM2/Phi unchanged. 4-module regression green.** |
| T5-CORRECTNESS-CERT-1 certify T5 WARP/AUTO vs CPU reference | **done (gated cert + docs; no runtime change). Added `T5MixedRuntimeCorrectnessCertTest` (directml-inference): (a) synthetic device cert `-Dt5.correctness.cert=true` — same tiny `.wdmlpack` through `T5Runtime.load` (reference) vs `T5Runtime.loadWarp` (WARP mixed), comparing greedy token ids + LM-head logits; ran on this host (RTX 5080/WARP): token ids identical (firstDivergent=none) for relu+gated configs, LM-head maxAbsLogitDiff=0.0 + top-1 identical → WARP dense-projection arithmetic is bit-exact vs reference (FP32 path; caveat: tiny weights collapse greedy to a constant token, so the 0.0 logit delta is the meaningful signal). (b) real-model cert `-Dt5.realModel=true` for t5-small/flan-t5-small/codet5-small/codet5-base-multi-sum — SKIPPED (no T5 artifact present locally; @EnabledIf disables cleanly). Forwarded `t5.correctness.cert`/`t5.realModel`/`t5.testModelDir*` toggles in directml-inference/build.gradle. **Verdict: D (data not yet sufficient — real-model package missing).** Synthetic arithmetic faithful, but no end-to-end real-model text parity measurable; product stance unchanged & honest: CPU = certified/recommended, WARP/AUTO = experimental/uncertified end-to-end. No guard/label change needed (existing NOTE already warns). Gemma/Qwen/SmolLM2 unchanged. 4-module regression green; gated cert green on this host.** |
| T5-PRODUCT-AUDIT-1 audit + honest T5/Flan-T5/CodeT5 workbench status | **done (audit/labels only; no T5 runtime change). All four curated T5 models (codet5-small, codet5-base-multi-sum, t5-small, flan-t5-small) are EXPERIMENTAL/runnable; not PLANNED, so no guard exemption exists or was needed. Routing honest: CPU→validated Java reference seq2seq runtime; WARP/AUTO→mixed path (`T5Runtime.loadWarp`: encoder+decoder+LM-head WARP boundaries) — dense projections (attention/FF + LM-head matvecs) on DirectML via shared WarpDenseProjection/MatMulNBitsKernel, norms/softmax/relative-bias on CPU reference; executes but not yet correctness-certified (surfaced as experimental, not finished-native); no Python; missing package→clear "Download/compile first". Added honest panel NOTE disclosing the mixed WARP/AUTO vs validated CPU routing (panel also prints exact executionMode). Fixed stale `T5Runtime` class javadoc ("API shell for the future T5 WARP runtime") and `UNSUPPORTED_MESSAGE` ("not implemented yet"→"not certified yet"; no longer thrown). Tests: `WorkbenchModelStatusAuditTest.t5FamilyIsRunnableByStatusWithHonestNotes`. Docs: workbench-model-status.md (T5 row + audit section). T5 not in SUPPORTED_MODELS.md. Gemma/Qwen/SmolLM2 unchanged. 4-module regression green.** |
| SMOLLM2-PRODUCT-AUDIT-1 audit + honest SmolLM2 workbench status | **done (audit/labels only; no SmolLM2 runtime change). Both SmolLM2 135M/360M are EXPERIMENTAL/runnable; removed the redundant `!smolLm2Model` PLANNED-guard exemption (runnable by status, like Qwen 0.5B). Routing honest: WARP→native DirectML/WARP (dense projections on D3D12 software rasterizer) with transparent CPU reference-runtime fallback (panel prints Runtime mode + Runtime fallback + reason); AUTO→hardware GPU else reference; CPU→reference; no Python; missing package→clear "Download → Convert". Fixed stale readiness wording in SummarizerPanel ("…the next WARP implementation step" → "WARP path prepared but not executable here; using the CPU reference runtime (<reason>)") — `SmolLM2NativeWarpExecutor` is the default; the "probe" executor ("kernels not implemented yet") is a `-Dsmollm2.warp.executor=probe` diagnostic. Refined the two SmolLM2 registry notes (native WARP + reference fallback, no Python). Tests: `WorkbenchModelStatusAuditTest.smolLm2IsRunnableByStatusWithHonestNotes`. Docs: workbench-model-status.md (SmolLM2 row + audit section). Gemma/Qwen unchanged. 4-module regression green.** |
| BUILD-OFFLINE-TEST-1 / BUILD-REGRESSION-BASELINE-1 (build infra) | **done. (OFFLINE-TEST-1) `directml-config` (java8Module, skipped by the root junit convention) declared an unversioned BOM-less `junit-platform-launcher` → `:directml-config:test` failed offline; fixed by mirroring the root `junit-bom:5.10.2` (launcher → cached 1.10.2, no version bump) → config tests run offline (55 tests, 4 classes, green; also validates the STATUS-2 GenerationModelRegistryTest). (BASELINE-1) Promoted `:directml-config:test` into the standard regression → now 4 modules (config + inference + encoder + workbench); documented in `CHATGPT_BUILD.md` ("Offline tests" + "Standard regression"; `chatgpt-build.sh`'s `clean build` already covers config — no new script). No online refresh needed. Rule: java8Module projects declare JUnit via the BOM, never an unversioned launcher.** |
| WORKBENCH-MODEL-STATUS-3 remove obsolete Gemma download-only remnants | **done (cleanup/lock-in; no downloader/runtime change). Removed the dead download/probe-only remnants: deleted `Gemma3DownloadLifecycle` (unused in production — `DefaultModelArtifactService` wires `Gemma3PackageLifecycle`; only a test referenced it); removed `ModelArtifactRow.gemmaDownloadOnlyStatusText` + its GEMMA3 branch and the `downloadOnlyCandidate` "Download only" label/tooltip (only ever matched the deleted lifecycle's reason). `NOT_SUPPORTED` now uniformly renders "Compiler missing" (behaviour-neutral for the remaining compiler-less families). Updated `Gemma3PackageLifecycle` javadoc (dropped the deleted-class reference). Tests: replaced the old `gemmaDownloadOnlyRow…` test with `gemma3UsesCompilerConvertLifecycleNotDownloadOnly` (Gemma raw-present+no-package → Convert/Download→Convert, never "Download only"); existing compiler-missing tests (SmolLM2/EMBEDDING) stay green. Gemma stays Download→Convert→READY → WARP/AUTO native; Qwen status unchanged; Python still only Gemma Backend=CPU. Regression 3 modules green.** |
| WORKBENCH-MODEL-STATUS-2 align Qwen status + drop the runnable-guard exception | **done. Qwen2.5-Coder-0.5B-it PLANNED→EXPERIMENTAL (runnable; native DirectML INT4 / model_q4f16.wdmlpack, no Python); registry note updated. Removed the `!qwenTestModel` PLANNED-guard bypass in SummarizerPanel — `isQwenTestModel` now only routes to QwenInferenceEngine (like the other families' routing), so no model is runnable via a hard-coded guard exception. Qwen 1.5B/3B stay PLANNED (clear "not executable yet"). Updated tests: config GenerationModelRegistryTest (Qwen 0.5B EXPERIMENTAL/runnable, only-0.5B-runnable among Qwen) + workbench WorkbenchModelStatusAuditTest (0.5B in runnable set + status assertion). Docs: workbench-model-status.md, SUPPORTED_MODELS.md, generation-api.md (Qwen 0.5B → experimental/runnable). Gemma unchanged; Python still only Gemma Backend=CPU. Regression 3 modules green (config tests not in the offline CLI set per the junit-launcher gotcha, but updated).** |
| WORKBENCH-MODEL-STATUS-1 audit all model families' workbench status | **done (audit-only; no runtime change). Verified Qwen/Gemma/T5/SmolLM2/Phi-3 dropdown/executability/backend routing + visible labels. Findings: no user-visible stale planned/probe/experimental residue (Gemma cleaned earlier; others honest), no "Runtime integration is planned" anywhere, Python only in Gemma Backend=CPU legacy (WARP/AUTO native, no silent fallback), executable families are runnable (not blocked by the PLANNED guard), genuinely-not-executable models stay PLANNED, Gemma Download-tab lifecycle is compiler-backed (Convert/READY, not download-only). Stale dead remnants noted (Gemma3DownloadLifecycle + gemmaDownloadOnlyStatusText, unreachable for Gemma — left for a later cleanup). Doc: `docs/workbench-model-status.md` (full status table). Lock-in test `WorkbenchModelStatusAuditTest` (4 invariants). Gemma unchanged. Regression 3 modules green.** |
| GEMMA-PRODUCT-CLOSEOUT finalize the Gemma product path | **done — docs closed (this section). Removed the last visible "probe" wording from the legacy CPU notes (SummarizerPanel) + the registry note; lock-in test `gemmaProductMetadataHasNoResearchOrPlannedWording` (note + native label carry no experimental/probe/planned/"until a native"). Verified WARP/AUTO native (no Python), Python only on CPU, clear product error messages, HF-token location documented. Paris unaffected (label/doc only). Regression 3 modules green.** |
| GEMMA-WARP-1 family shell + config/inspect | **done** (7b1dd3c) |
| GEMMA-WARP-2 tokenizer + chat template | pending |
| GEMMA-WARP-3 wdmlpack compiler shell | pending |
| GEMMA-WARP-4 CPU reference math | **done** (bd8fbb4) |
| GEMMA-WARP-8/9 reference weights + full forward | **done** (4b028fe) |
| GEMMA-WARP-9 real-model parity vs transformers | **done — PASS** (next token 9079 " Paris") |
| GEMMA-WARP-3 wdmlpack compiler + package | **done** (137f0f5) |
| GEMMA-WARP-7 attention layout + window mask (device-free) | **done** (0d83fbd) |
| GEMMA-WARP-10 reference greedy generation | **done** (036493b) |
| GEMMA-WARP-2 tokenizer + chat template | **done — exact transformers match** |
| GEMMA-WARP-5a WARP zero-centered RMSNorm + QK-Norm kernels | **done — GPU-validated vs reference** |
| GEMMA-WARP-6 WARP GeGLU + GELU-tanh MLP | **done — GPU-validated vs reference** |
| GEMMA-WARP-7a attention layout + RoPE reference (device-free) | **done** (layout 0d83fbd, RoPE this slice) |
| GEMMA-WARP-7b WARP RoPE + attention-scores primitives | **done — GPU-validated vs reference** |
| GEMMA-WARP-8 WARP single-layer (softmax + value + full attention) | **done — GPU-validated vs reference** |
| GEMMA-WARP-9 WARP full prefill (all layers + embedding + tied LM head) | **done — real model top-1 " Paris" on GPU** |
| GEMMA-WARP-10a WARP KV cache + single-token decodeNext | **done — real decode == full recompute on GPU** |
| GEMMA-WARP-10b WARP greedy generate + stop + streaming | **done — real " Paris." multi-step on GPU** |
| GEMMA-WARP-11 workbench native flag (-Dgemma.runtime=native-warp) | **done — native path " Paris." in the runner** |
| GEMMA-WARP-12 perf/heap measurement (native-warp vs external) | **done — measured; verdict WAIT** (see `gemma3-warp-runtime-performance.md`) |
| GEMMA-WARP-13a heap-light product weight load | **done — on-heap delta 0 MB at load, still " Paris"** |
| GEMMA-WORKBENCH-CONVERT-1 UI Convert → model_gemma3.wdmlpack | **done — Convert enabled + produces the package the runtime finds** |
| GENERATION-STREAMING-1 global streaming switch + Gemma native-warp streaming | **done — default streaming, Gemma streams " Paris." live** |
| GEMMA-WARP-13b-1 submit/fence/readback instrumentation | **done — measured: ~1140 submits + 344 readbacks per decode token** |
| GEMMA-WARP-13b-2 GPU-resident buffer/execution-context seam | **done — resident kernels, attn-chain 4→1 readbacks, parity** |
| GEMMA-WARP-13b-3a resident decodeStep wiring | **done — real readbacks/token 344→37, still " Paris"** |
| GEMMA-WARP-13b-3b submit/fence-coalescing | **done — real fenceWaits/token 834→93, submits 834→454, still " Paris"** |
| GEMMA-WORKBENCH-PROFILING-1 profile output + runtime UI | **done — Gemma runtime UI-selectable (no JVM flag), detailed phase/WARP-counter profile, 'Show runtime profile' toggle** |
| GEMMA-WARP-13b-4 resident/batched as native-warp product path | **done — native-warp defaults to resident; real profile 454 submits / 93 fenceWaits / 37 readbacks per decode token, still " Paris"** |
| WORKBENCH-CLEANUP-STREAMING-ONLY-1 remove extra UI controls | **done — only 'Streaming output' remains; Gemma native chosen by Backend=WARP; profile is a `-Ddirectml.generation.profile` debug path; `-Dgemma.runtime` no longer drives product logic** |
| GEMMA-WARP-13c GPU-resident KV cache | **done — per-layer GPU K/V cache, no per-layer readback; real readbacks/token 38→1, fenceWaits/token 97→21, still " Paris"** |
| GEMMA-WARP-13d command-list coalescing | **done — UAV dispatches coalesced per layer; real submits/token 418→220, fenceWaits 21 (unchanged), decode ~442→294 ms/token, still " Paris"** |
| GEMMA-WARP-13e batched prefill | **done — whole prompt in one pass (batched projections + MLP); real prefill submits 1417→261 (6-tok), top-1 still " Paris", decodeNext after batched works** |
| GEMMA-WARP-13f warm/cold profiling + shader warm-up | **done — Gemma3WarpWarmup + gated cold/warm steady-state profile; prefill submits ~constant in prompt len (261/280/355 @ 6/25/100), decode warm ~250 ms/token; finding: decode submit-bound (→ matvec coalescing next), long prefill compute-bound (per-position attention)** |
| GEMMA-WARP-13g projection/matvec submit reduction | **done — matvec recorded into the layer command list (DML numerics unchanged); real decode submits/token 220→21, fences 21, readbacks 1, top-1 " Paris" identical. Finding: decode avg/token ~250 ms unchanged → decode is compute-bound (matvec FLOPs on WARP), not submit-bound** |
| GEMMA-WARP-14a matvec compute benchmark + decision | **done — DML-GEMM fastest (~20 ms matvec/decode-token) vs custom WARP-FP32 (~130) / INT4 (~145); DECISION: keep DML. Bigger finding: matvec is only ~20 ms of ~250 ms decode → bottleneck is the many small element/attention dispatches+barriers, not matvec → next lever is kernel fusion** |
| GEMMA-WARP-14b fuse attention (scores+softmax+value) | **done — Gemma3WarpFusedAttentionContextKernel (flash-attention online softmax, 3 dispatches→1) + dispatches/uavBarriers counters. Real decode/token dispatches 416→380, uav barriers 435→399; submits/fences/readbacks 21/21/1 unchanged; top-1 " Paris" identical. decode avg/token noisy/flat (fused trades 3 dispatches for 1 heavier; ~equal short, better long context)** |
| GEMMA-WARP-14c performance-ceiling doc + decision | **done — `docs/gemma3-warp-performance-ceiling.md`: history + bottleneck (fixed per-dispatch/barrier overhead of ~380 tiny dispatches on the WARP CPU rasterizer; native runtime inits `directml`=default adapter, WARP only because it is adapter 0 here). Recommendation: measure on a hardware GPU (GEMMA-AUTO-GPU-1) before more WARP fusion. No code/runtime change** |
| GEMMA-PRODUCT-2 polish the Gemma product status / visible UI residue | **done (no runtime/downloader change). Audited where research wording is visible: the registry `EXPERIMENTAL` status is NOT shown (dropdown lists modelId only, no status tooltip); the only visible "experimental" was the runtime-mode label `Gemma3RuntimeMode.NATIVE_WARP` = "native-warp-experimental", shown as "Runtime mode: …" + in the profile report → renamed to **"native-warp"** (product label) + javadoc updated (native = product path for WARP/AUTO, not flag-gated). Fixed the stale SummarizerPanel comment ("'Gemma runtime' control"). Registry status kept EXPERIMENTAL internally (consistent with Phi-3/SmolLM2/T5; not user-visible) — locked by a test that the native label has no "experimental"/"probe" wording. Verified all product error messages are clear (missing model_gemma3.wdmlpack → Download→Convert; gated HF → accept terms + read token; AUTO no hardware → use WARP / adapterIndex; WARP init errors surfaced with cause). WARP/AUTO stay native (no Python); Python only on Backend=CPU. HF token in `%APPDATA%/.directml/download-overrides.json`. Tests: `gemmaNativeRuntimeLabelHasNoResearchWording` + profile-report literal updated. Paris (9079) unaffected (label/comment/metadata only; product-path smoke green in PRODUCT-1). Regression 3 modules green.** |
| GEMMA-PRODUCT-1 wire/verify the Gemma product path in the workbench | **done (audit + stabilize; the path was already wired). Verified: `google/gemma-3-270m-it` is EXPERIMENTAL in `GenerationModelRegistry` → selectable in the Summarizer dropdown + runnable (not blocked by the PLANNED guard). `SummarizerPanel` routes Backend=WARP→native DirectML on the WARP software adapter (no Python), Backend=AUTO→native on the optional hardware adapter (clear error if none, via AUTO-GPU-1), Backend=CPU→external Python probe only; missing `model_gemma3.wdmlpack` fails clearly (no silent Python fallback). Download/import: gated HF model handled via per-model token in `%APPDATA%/.directml/download-overrides.json` (`huggingFaceTokens`, edited in the download access dialog/gear), sent only to huggingface.co; 401/403 → clear `gatedAccessHint`. UI lean: Backend WARP/AUTO only, no runtime selector / no visible profile toggle / no research mode. Stabilization: fixed the stale registry note (was "until a native Java/WARP Gemma runtime exists"); added `gemma3ItIsSelectableAndRunnableAsAProduct` regression. Product-path Paris (9079) green via `Gemma3NativeWarpRuntime` (the SummarizerPanel runtime class, `.wdmlpack` load) on WARP + HARDWARE. Regression 3 modules green.** |
| GEMMA-BF16-PACK-CLOSEOUT finalize the BF16 host-memory line | **done (docs/stabilization only; no code change, no PACK-4). Ceiling doc §16: BF16 packaging is a host-RAM optimization not a speed one — payload was already BF16 in `.wdmlpack` (no format change), FP32 only at load, host weights now retained BF16, FP32 only transient for upload, device buffers stay FP32 (decode floor), ~511 MB system-RAM saved, decode unchanged, Paris green, QKV/GateUp + BF16 host-views active. Verified: all BF16 real-model probes gated (`realModel`+`bf16Probe`), non-gated BF16 tests are pure-CPU, default run light; documented the two-WARP-sessions-in-one-JVM DEVICE_REMOVED caveat (product holds one); BF16 main path adds no logs (runtime silent), no UI/selector/research mode; product path unchanged (WARP native no-python / AUTO optional / CPU external-python only). Regression 3 modules green. Gemma WARP line complete — speed at fp32 ceiling (§1–§12) + host memory minimized (§13–§16)** |
| GEMMA-BF16-PACK-3 layer projections kept BF16 in host RAM (line closed) | **done (memory only; no `.wdmlpack` format change, no GPU/GEMM change, no speed change — PACK-3 touches only load). QKV/GateUp fusion (WARP-16) preserved. New `WarpWeightSource.ofLazyFp32`/`resolveFp32LittleEndian` (lazy transient FP32, no retained FP32); `Gemma3WarpLayerWeights.ofBf16Projections` (7 `Gemma3Bf16WeightView`, lazy-FP32 sources); `WarpDenseProjection` fused/single builders use resolve; `loadWarpWeightsHeapLight` retains BF16 projections for a BF16 model. Decoder-only family unaffected (still uses `fp32LittleEndian()`). Measured: projection host RAM **382.5→191.3 MB (−191.3)**; cumulative w/ PACK-2 **1022.5→511.3 MB (−511.3)**; load 320 ms; decode 571 ms (unchanged); byte-identical projection parity (layer 0, all 7); Paris green. Tests: `WarpWeightSourceTest` (lazy), `Gemma3Bf16ProjectionLoadProbeTest` (gated). Regression 3 modules green. **BF16-pack line CLOSED** — retained host weights now BF16 end-to-end; remaining FP32 is only device buffers (fp32-GEMM floor) + norms. See ceiling doc §15** |
| GEMMA-BF16-PACK-2 tied embedding/LM-head kept BF16 in host RAM | **done (memory only; no `.wdmlpack` format change, no GPU/GEMM change, no speed promise). New `Gemma3Bf16WeightView` (retained BF16 host view: `decodeRowScaled` per-row widen for the lookup, `inflateToFp32` one-off for the LM-head upload); `Gemma3WarpWeights.ofBf16Embedding` + domain methods `embedScaled`/`buildLmHead` (session + forward pass no longer branch on storage form); `Gemma3RuntimePackage.loadWarpWeightsHeapLight` retains BF16 for a BF16 payload. Measured (real model, WARP): retained embedding host RAM **640→320 MB (−320 MB)**; embedding load 1690→127 ms (~13× faster, bulk copy); decode avg/token 570.6→577.1 ms (+1.1%, within noise); row decode for ids 0/1/9079/last/random byte-identical; Paris green. Only the tied block; device buffer stays FP32. Regression 3 modules green. BF16-PACK-3 (layer projections, ~191 MB more) only if host RAM still a constraint. See ceiling doc §14** |
| GEMMA-BF16-PACK-1 lossless 16-bit weight storage (investigation only) | **done (memory/packaging investigation, NOT speed; no format/product-path change). `Gemma3Bf16PackFeasibilityProbeTest` (`-Dgemma.warp.bf16Probe=true`). Findings: all matmul weights are BF16 in safetensors; the `.wdmlpack` payload = verbatim safetensors → **already BF16 on disk (no disk saving)**. FP32 arises only at runtime load (`Gemma3WeightBufferView.decodeFp32LittleEndian` widens BF16→FP32 into retained off-heap ByteBuffers; device buffers FP32 for the fp32 DML GEMM). Measured: matmul weights BF16 511 MB vs FP32 1022 MB → up to ~511 MB host/system-RAM saving if kept BF16 + inflated at use (embedding lookup 320 MB is the biggest, must-retain item). Device FP32 is the fp32-GEMM floor; reducing it needs an fp16 operator (out of scope, no speed benefit §11). BF16→FP32 inflate verified lossless; Paris green. Decision: NO format/manifest change needed (already BF16). GEMMA-BF16-PACK-2 recommended ONLY as a deliberate host-memory optimization (runtime keeps BF16, inflate at use); deferrable if RAM not a constraint. See ceiling doc §13** |
| GEMMA-WARP-CLOSEOUT finalize + stabilize the WARP speed line | **done (no new optimization). Final docs: ceiling doc §12 (final verdict, trajectory, gated profiling/probe property table, verified product path). Verified: all real-model probes gated (`gemma.warp.realModel` + a per-probe opt-in); default test run light; product path unchanged (WARP→explicit software adapter no-Python, AUTO→hardware optional, CPU→external python only, no runtime selector / no new UI switch). Stabilization: lowered the per-session "Gemma projection fusion" log INFO→DEBUG so the normal runtime is quiet. WARP = stable CPU-only baseline ~500 ms/token at fp32 compute ceiling; AUTO/GPU optional; quantization = separate memory decision. Regression :directml-inference :directml-encoder :directml-workbench green** |
| GEMMA-WARP-19 quantization/precision feasibility (decision only) | **done (decide only; no production quantization, no format/runtime change). `Gemma3WarpPrecisionFeasibilityProbeTest` (`-Dgemma.warp.quantProbe=true`, WARP adapter): fp16/bf16 LM-head weight rounding through the existing FP32 GEMM → Top-1 identical, deviation **0** (model is stored BF16 — verified — so 16-bit storage is lossless, halves weights 640→320 MB). Paris green. **Decision D: quantization does NOT help WARP speed** — WARP is a compute-bound CPU rasterizer at its fp32 DML-GEMM ceiling; INT4 already measured ~7× slower (14a), fp16/INT8 GEMM not wired and would not speed compute-bound WARP. 16-bit *storage* is a free *memory* win but a separate product decision, not a speed lever. Stop the WARP speed-optimization line here; WARP = stable CPU-only product path (~500 ms/token), AUTO/GPU = optional accelerator. GEMMA-WARP-20 should NOT be a speed-quantization slice. See ceiling doc §11** |
| GEMMA-WARP-18 LM-head DML-GEMM shape probe | **done (measure only; product path unchanged). `Gemma3WarpLmHeadShapeProbeTest` (`-Dgemma.warp.lmHeadProbe=true`, WARP adapter) compared GEMV vs GEMM(M=1) vs row-blocked, all verified identical full-vocab Top-1, Paris green. Result: baseline GEMV (matvecResident, ~265 ms) is OPTIMAL — GEMM M=1 is 3.3× slower (874 ms), row-blocking neutral/worse (4×+0.7%, 16×+5.9%), 0 memory overhead but 0 gain. WARP decode is at its fp32 DML-GEMM compute ceiling (~0.6 GMAC/s; LM head 168M MACs/265ms). Recommendation: do NOT change the LM head; further WARP speedup needs FLOP reduction (quantization, out of scope) else GPU/AUTO. See ceiling doc §10** |
| GEMMA-WARP-17 projection-fusion hardening + measured next-bottleneck breakdown | **done (no new optimization). Hardening: parity tests (fused QKV/GateUp == separate), `WarpGpuBuffer.slice` edge-case tests, `mlpBatched` per-position fallback when batch unsupported. Measurement: opt-in `WarpGroupProfiler` + `WarpExecutionContext.mark/endGroup` (`-Dgemma.warp.groupProfile=true`, normal runtime untouched). MEASURED WARP decode (real model, profiled total ≈ real, 531≈522 ms/token): decode is matmul-bound — **lm-head 52%, gate+up 17%, qkv 7.8%, down 7.6%, o 5.2% (5 GEMMs ≈ 90%); all element kernels ≈ 10%**. Paris green. Recommendation: GEMMA-WARP-18 targets the tied LM-head GEMM (52%), then the MLP GEMMs; do NOT pursue small-element fusion. See ceiling doc §9** |
| GEMMA-WARP-16 fuse QKV + GateUp projection groups | **done — runtime-prepacked fusion: q/k/v → one fused QKV DML-GEMM (`[attnDim+2·kvDim, hidden]`, output sliced into Q/K/V zero-copy `WarpGpuBuffer.slice` views), gate/up → one fused GateUp DML-GEMM (`[2·intermediate, hidden]`, `[gate|up]` feeds the single-buffer GeGLU). `WarpDenseProjection.fromFusedWeightSources` → `MatMulNBitsKernel.fromFusedFp32ByteBuffers` (heap-light; packed at load only — no `.wdmlpack` change). DML-GEMM kept (no custom/INT4 matvec). Byte-identical (float[] oracle slices too; MLP/layer parity tests green). Measured: dispatches/token 344→290, uav barriers 363→309 (−54 = 3/layer×18), submits/fences/readbacks 21/21/1; **WARP decode ~600→~525 ms (~15–20% faster)**, totals A/B/C 15339/16775/23287→13189/13509/17031 ms; GPU ~210→~150 ms. QKV+GateUp fused active=true. Top-1 " Paris" identical WARP+HARDWARE. The real WARP lever is GEMM shape/count, not element-dispatch count. See ceiling doc §8** |
| GEMMA-WARP-15 per-layer dispatch breakdown + norm+add fusion | **done — `Gemma3DecodeDispatchBreakdown` (per-token/per-layer, validated vs measured 380). Identified the post-attn/post-ff RMSNorm+residual-add as the largest fuseable small-dispatch group; fused into `Gemma3WarpFusedNormAddKernel` (`ZERO_CENTERED_RMSNORM_ADD_HLSL`), used at both post-norm sites in decodeStepResident + forwardPrefillBatched. Byte-identical to norm-then-add (unit test vs CPU reference). Measured: dispatches/token 380→344, uav barriers 399→363, submits/fences/readbacks 21/21/1 unchanged; decode wall-clock flat (WARP ~600 ms, RTX 5080 ~210 ms). Top-1 " Paris" identical WARP+HARDWARE. Reinforces 14b: dispatch count is no longer the WARP lever. See ceiling doc §7** |
| GEMMA-AUTO-GPU-1 native Gemma on hardware adapter vs WARP | **done — `WindowsBindings.AdapterMode` (WARP/HARDWARE/DEFAULT) decoupled from backend `directml` (DML-GEMM kept, per 14a); `DxgiBindings.getAdapterDesc1` adapter identity; selectHardwareAdapter skips software with clear no-hardware message. Workbench: Backend=WARP→native(WARP adapter), Backend=AUTO→native(HARDWARE adapter) — no external-Python on the Gemma product path. Profile carries adapter desc/software. Gated `Gemma3AutoGpuProfileTest` (real model) measured on RTX 5080: decode ~217 ms/token (HARDWARE) vs ~600 ms/token (WARP) = ~2.7× with identical 380 dispatches/21 submits/399 barriers per token → confirms fixed per-dispatch CPU-rasterizer overhead. Top-1 " Paris" identical on both. See `gemma3-warp-performance-ceiling.md` §6** |
| GEMMA-WARP-10 WARP decode session + KV cache | open — depends on WARP kernels |
| GEMMA-WARP-11 workbench native flag | open — depends on tokenizer + WARP |
| GEMMA-WARP-12 perf/heap comparison | open — depends on WARP |

## What works now (verified, device-free, no Python)

A complete **native CPU Gemma 3 runtime** in Java, numerically correct vs HF transformers:
`config.json` + `model.safetensors` → `Gemma3WdmlPackCompiler` → `.wdmlpack` →
`Gemma3RuntimePackage.loadReferenceWeights()` → `Gemma3ReferenceForwardPass` /
`Gemma3ReferenceGenerator` (greedy). Real-model parity PASSES (next token 9079 " Paris"). 24 gemma
tests green (1 gated parity ran locally). This CPU path is the **WARP parity oracle**, not the product
path.

**Milestone:** the device-free CPU reference is numerically correct against HF transformers
(`Gemma3RealModelParityTest`, gated): for "The capital of France is" (ids [2,818,5279,529,7001,563])
the Java reference greedy next token == transformers argmax == 9079 (" Paris"). The Gemma math
(zero-centered RMSNorm, QK-norm, dual-theta RoPE, GQA, sliding window, GeGLU, sandwich norms, tied LM
head) is therefore the trustworthy WARP parity oracle.

## Open points / blockers (resolve at the end)

- **GPU/WARP execution IS available on this host (`WHERETHEHEARTIS`).** The earlier "no device in the
  sandbox" assumption no longer holds for this machine: the existing `SmolLM2NativeWarpExecutorTest`
  runs (5 tests, 0 skipped, `backend=warp`), and the new `Gemma3WarpNormKernelTest` runs all 7 GPU
  cases. WARP-kernel slices are therefore validated on the real device, gated on
  `WindowsBindings.isSupported()` so they skip (not fail) on a CI box without an adapter. Strategy
  unchanged: the CPU reference (numerically correct vs HF/transformers) is the parity oracle; each
  WARP kernel is mirrored against it.

- **GEMMA-WARP-5a norm kernels — done, GPU-validated.** `Gemma3WarpNorms` (HLSL), `Gemma3WarpRmsNormKernel`
  (zero-centered RMSNorm over a whole vector) and `Gemma3WarpQkNormKernel` (per-head RMSNorm over
  `head_dim`, all heads in one dispatch). Both are the GPU mirror of `Gemma3ReferenceMath.rmsNormZeroCentered`
  and **scale by `(1 + weight)`** — kept strictly separate from the Phi-3/Qwen `RMSNORM_HLSL` (which
  scales by `weight`); those shaders are untouched. Validated on the real device against the verified
  CPU reference for small shapes, the real hidden=640 and head_dim=256 (incl. numHeads=1 GQA k-head),
  multiple eps, and the zero-weight identity-scale case. Tolerance **abs 1e-4 + rel 1e-4** (GPU float
  vs reference double sum-of-squares). `head_dim` is supplied by the caller, never derived as
  `hidden/heads`. Still validation building blocks (per-call upload/readback), not yet the fused
  product path.

- **GEMMA-WARP-6 GeGLU + GELU-tanh MLP — done, GPU-validated.** `Gemma3WarpActivations` (HLSL),
  `Gemma3WarpGeluTanhKernel`, `Gemma3WarpGeGluKernel` (fused `gelu_tanh(gate)*up`), and `Gemma3WarpMlp`
  composing the three matmuls over the **shared `WarpDenseProjection`** + the GeGLU kernel
  (`down_proj(gelu_tanh(gate_proj·x) * up_proj·x)`; the pre/post feedforward sandwich norms stay in the
  5a norm kernels, not in this block). Gemma's **GELU-tanh GeGLU** is kept strictly separate from the
  Qwen/SmolLM2 SiLU SwiGLU; the decoder-only SwiGlu kernels are untouched. Validated on the real device
  against the verified reference: GELU-tanh on signed/zero/tail values and a 2048-wide random vector,
  GeGLU on the real intermediate (incl. zero-gate→0), and the full MLP on a small shape and the real
  hidden=640/intermediate=2048 shape. Tolerance **abs 1e-4 + rel 1e-4** for the element-wise activation
  kernels; **abs 1e-3 + rel 1e-3** for the full MLP (three float matmuls accumulate in a different order
  than the reference dot). `MatMulNBitsKernel.fromDequantizedWeights` uploads the full FP32 matrix
  (no re-quantization), so this is a true FP32 parity — the real-shape test uses realistic small weight
  magnitudes (the regime a pre-feedforward-normed input actually produces). ByteBuffer norm-weight
  upload intentionally skipped (norm/activation vectors are tiny; heap-light matters for the big
  projections + tied embedding/LM-head).

- **GEMMA-WARP-7 RoPE + local/global attention layout — done.** Split into 7a (device-free) and 7b
  (WARP). **7a:** the local/global layout (`Gemma3AttentionLayout`) was already shipped in 0d83fbd
  (GQA mapping, full layers 5/11/17, dual theta, sliding-window + causal visibility); this slice adds
  the device-free `Gemma3RoPE` (rotate-half across heads, the parity oracle) plus `Gemma3RoPETest`
  rounding out the checklist (all-18-layer full/local classification, **explicit head_dim=256** — never
  `hidden/heads`, GQA with >1 kv head, window+causal). **7b:** `Gemma3WarpAttention` (HLSL) +
  `Gemma3WarpRoPEKernel` (rotate-half RoPE over packed heads) and `Gemma3WarpAttentionScoresKernel`
  (scaled masked `QK^T` for one query position; GQA `kvHead=head/groupsPerKv`; the visible range
  `[firstValid, queryPos]` comes from `Gemma3AttentionLayout`, so local/global + causal are applied
  without imitating any other family; masked keys get the `SCORE_SENTINEL = -1e30` a softmax drops).
  Validated on the real GPU vs the verified reference: RoPE on a small shape and the real head_dim=256
  at **both** thetas (1e4/1e6) and several positions (pos=0 identity); scores on a full layer
  (firstValid=0), a local layer with a biting sliding window, and a 4-heads/2-kv GQA case. Tolerance
  **abs 1e-4 + rel 1e-4**. Still primitives (per-call upload/readback), not yet the fused single-layer
  step — that is GEMMA-WARP-8 (softmax + AV + o_proj wiring).

- **GEMMA-WARP-8 single-layer — done, GPU-validated.** `Gemma3WarpSoftmaxKernel` (numerically-stable
  row-wise softmax; masked sentinels exp to 0) + `Gemma3WarpAttentionValueKernel`
  (`context = prob·V`, GQA `kvHead=head/groupsPerKv`) complete the attention chain; `Gemma3WarpLayer`
  composes the full Gemma layer from the validated blocks —
  `input RMSNorm → q/k/v proj → QK-norm → dual-theta RoPE → GQA attention (local/global sliding-window
  + causal via Gemma3AttentionLayout) → o_proj → post-attention RMSNorm → +residual → pre-ff RMSNorm →
  GeGLU MLP → post-ff RMSNorm → +residual`. `Gemma3WarpLayerWeights` is the WARP-side per-layer holder
  (`from(Gemma3ReferenceWeights.Layer)`); a new `Gemma3ReferenceForwardPass.runLayer(state, layer)`
  exposes the single-layer oracle. Validated on the real GPU vs the reference for a tiny local layer
  (biting window), a tiny full layer, a 4-heads/2-kv GQA layer, and the **real head_dim=256 geometry**
  (hidden=640, inter=2048) both full and local. Tolerance **abs 2e-3 + rel 2e-3** — looser than the
  element-wise kernels because a layer chains many float kernels against a reference that accumulates
  RMSNorm/RoPE/softmax in double; small realistic weight magnitudes (the normed-input regime). Still a
  building-block composition (CPU readback between kernels), not the fused single-submission pipeline.

- **GEMMA-WARP-9 full prefill — done, real-model GPU-validated.** `Gemma3WarpEmbedding` (lookup
  ×sqrt(hidden), reads only the prompt rows), `Gemma3WarpLmHead` (tied to `embed_tokens`, one GPU upload;
  **heap-light FP32 ByteBuffer** seam + a `float[]` convenience), `Gemma3WarpWeights` (whole-model
  weights, float[] or direct-ByteBuffer embedding), `Gemma3WarpKernels` (shared stateless kernels built
  once across all layers), and `Gemma3WarpForwardPass` (`token ids -> embedding -> 18 layers -> final
  RMSNorm -> tied LM head -> logits`; LM head built lazily so 9a needs no 256k matrix). Validated on the
  real device: synthetic full-prefill exact parity (incl. tied LM head + the heap-light ByteBuffer path
  == float[] path), real-model first-4-layer element parity on BF16 weights, and the **full 18-layer
  real prefill top-1 == 9079 (" Paris")** — matching transformers/the reference. Tolerance abs 3e-3 +
  rel 3e-3 for the synthetic hidden/logits; **top-1 is the asserted metric for the real model** (no
  full 256k-logit identity claim).
  - **Two real-model issues fixed here (both float-vs-double, not logic):**
    1. **GELU-tanh NaN.** The HLSL `tanh` intrinsic (exp-based) overflows float to `inf/inf = NaN` once
       its argument exceeds ~88; Gemma's large real activations (e.g. gate ~= 13 -> arg ~= 88) hit this,
       while synthetic small values never did and the reference uses double `Math.tanh`. Fixed by clamping
       the GELU-tanh argument to +/-20 (tanh saturates to +/-1 there to float precision — mathematically
       exact, overflow-safe). The WARP-6 activation tolerances are unaffected (their args stay < 20).
    2. Shared stateless kernels (`Gemma3WarpKernels`) instead of ~7 per layer — bounded resident
       kernel/PSO/command-list count at 18-layer scale.
  - **Heap note:** the WARP LM head/embedding uses a direct FP32 ByteBuffer decoded from the SafeTensors
    payload (no 164M-weight host `float[]`); the layer projection weights are still host `float[]` (the
    ByteBuffer projection seam exists and can be wired in the product slice). The reference path keeps
    its FP32 `float[]` embedding (parity only).
  - **For GEMMA-WARP-10 (decode session) still missing:** a KV cache (append per step; local layers only
    need a windowed cache), a single-token decode step reusing cached k/v, the generate/streaming loop +
    EOS, and (perf) a fused single-submission pipeline instead of per-call upload/readback.

- **GEMMA-WARP-10a KV cache + single-token decode — done, real-model GPU-validated.** `Gemma3WarpKvCache`
  (per-layer flat `[position][kvDim]` k/v, grows on append; stores the **full** history and applies the
  local/global sliding window at *read* time via `Gemma3AttentionLayout.firstValidKey` — windowed
  eviction is a later perf concern), `Gemma3WarpLayer.decodeStep` (single new token: rmsnorm → q/k/v →
  QK-norm → RoPE at `pos` → append to cache → attention over the visible cached range → o_proj → residual
  → GeGLU MLP → residual, reusing the shared kernels + the layer's projections), and
  `Gemma3WarpDecodeSession` (`prefill` then `decodeNext`; **prefill and decode share the one per-token
  path**, so token `t` attends to `[firstValid(t), t]` exactly as a full-sequence causal pass — prefill's
  last-token logits equal `Gemma3WarpForwardPass`). Tied LM head lazy; reuses `Gemma3WarpKernels`.
  Validated on the real device: synthetic prefill+decodeNext exact parity vs the CPU reference for a
  full-history config **and** a small-sliding-window config (the local window bites as the cache grows),
  cache length grows correctly, and on the real Gemma 3 270M **prefill top-1 = 9079 (" Paris")** and
  **decodeNext(9079) top-1 = 236761**, matching the full-recompute WARP forward pass (the cache path ==
  re-running the whole sequence). Tolerance abs 3e-3 + rel 3e-3 (synthetic logits); top-1 is the asserted
  real-model metric.
  - **For GEMMA-WARP-10b (full generation) still missing:** the multi-step greedy generate loop driving
    `decodeNext` repeatedly, EOS/end-of-turn stopping + the `Gemma3Tokenizer` decode of the produced ids,
    an `IntConsumer`/streaming callback, and (perf, separate) a fused single-submission pipeline +
    windowed-eviction KV cache + the ByteBuffer projection-weight path. No workbench wiring / runtime
    default switch yet.

- **GEMMA-WARP-10b greedy generation — done, real-model GPU-validated.** `Gemma3WarpGenerator`
  (prefill → select → decodeNext → repeat) over the decode session, with `Gemma3TokenSelector` (greedy =
  argmax, sampling-ready seam), `Gemma3StopTokenPolicy` (stop-id set; `ofEos` / `ofEosAndEndOfTurn` —
  Gemma instruct ends a turn with `<end_of_turn>`, not always `<eos>`), `Gemma3GenerationRequest`
  (promptIds + maxNewTokens) and `Gemma3GenerationResult` (`generatedTokenIds`/`fullTokenIds` both
  stop-token-free, `FinishReason` STOP_TOKEN/MAX_TOKENS, prompt/output counts). The stop token ends
  generation, is excluded from the visible output, and is **not** streamed — the `IntConsumer` receives
  exactly `generatedTokenIds`. Validated on the real device: synthetic multi-step greedy equals a manual
  greedy loop over the CPU reference; max-tokens and stop-token (stop on the first token → empty output,
  not streamed) contracts hold; streaming == visible result; and the real Gemma 3 270M generates
  **"The capital of France is" → ids [9079, 236761, 108] = " Paris."** (first token " Paris", MAX_TOKENS
  at 3, streaming consistent) — no text-quality claim beyond the first token + stable execution.
  - **For GEMMA-WARP-11 (workbench) still missing:** a runtime entry point that tokenizes via
    `Gemma3Tokenizer`/`Gemma3ChatTemplate`, runs `Gemma3WarpGenerator`, and decodes to text behind the
    workbench's model/runtime descriptor + a `-Dgemma.runtime=native-warp` flag (external-python stays
    default until proven); plus the heap-light product weight load (ByteBuffer projections, wdmlpack
    payload) and the perf items above. No runtime default switch / downloader / .wdmlpack change here.

- **GEMMA-WARP-11 workbench native flag — done.** `Gemma3RuntimeMode` (`-Dgemma.runtime`: `external`
  default / `native-warp`), `Gemma3NativeWarpRuntime` (tokenizer.json → `Gemma3Tokenizer`/optional
  `Gemma3ChatTemplate` → weights from the compiled `model_gemma3.wdmlpack` via `Gemma3RuntimePackage` →
  `Gemma3WarpGenerator` → detokenized text). `SummarizerPanel.runGemma3Generation` branches on the mode:
  default keeps the **unchanged** external Python path; `native-warp` runs the native runner and logs
  `Runtime mode: native-warp-experimental`, model id/dir, `Backend: WARP`, `Package: model_gemma3.wdmlpack`,
  prompt/output tokens, finish reason. **Missing package with an explicit native flag fails clearly**
  (`"Gemma native WARP requires a compiled .wdmlpack package (...). Use Download/Convert first or run the
  Gemma compiler."`) and does **not** fall back to Python; likewise a missing device fails clearly.
  Validated: runtime-mode parsing (default external, native-warp on flag), the missing-package message is
  actionable, and a real GPU smoke that compiles a `.wdmlpack` and generates **"The capital of France is"
  → " Paris."** through the native runner (`Gemma3NativeWarpRuntimeTest`). external default unchanged,
  not switched. Weights still load through the package's `float[]` reference path (heap) — correct, not
  yet heap-light; the ByteBuffer/heap-light product load + perf remain GEMMA-WARP-12.
  - **Perf note (rough, no optimization claim):** the per-call upload/readback building-block path makes
    the native runner clearly slower than a fused pipeline would be; the gated tests run in the tens of
    seconds (model load + prefill + a few decode steps on the WARP software rasterizer). Real numbers and
    any speedup are GEMMA-WARP-12.
- **GEMMA-WARP-2 Tokenizer — RESOLVED (done).** Native `Gemma3Tokenizer` reads `tokenizer.json` only
  (no `tokenizer.model`, no SentencePiece DLL, no Python): normalizer space→`▁`, BPE over the whole
  normalized string (the `Split(" ")` pre-tokenizer is a no-op post-normalization), byte_fallback to
  `<0xNN>`, added/special tokens carved out before BPE, `<bos>` prepended. Java ids match HF
  transformers **exactly** for all fixtures (English, German umlauts, Natural/ADABAS, Java, empty,
  unicode/byte-fallback) and the chat template; decode round-trips. Findings: **`tokenizer.json` alone
  suffices** (no `tokenizer.model` needed); byte-fallback **is** required (and implemented); metaspace
  normalizer correct; no leading-`▁` prepend at start. The full native chain
  **text → Java tokens → Java forward → Java decode = "Paris"** is verified (`nativeTextToTextProducesParis`).
  → A text→text workbench native path is now possible (pending the WARP runtime for speed).

- **WARP kernels (GEMMA-WARP-5/6/8/9/10, GPU-blocked here):** no DirectML/D3D12 device in this build
  sandbox, so on-GPU numerical validation cannot run (mirrors `SmolLM2NativeWarpExecutorTest`, gated on
  `WindowsBindings.isSupported()`). The CPU reference + `Gemma3AttentionLayout` give a ready parity
  oracle. Implementation plan when run on a Windows+GPU host: reuse `WarpDenseProjection`/
  `MatMulNBitsKernel` + the D3D12 ByteBuffer upload for q/k/v/o/gate/up/down and the LM head; add
  **new** Gemma kernels for zero-centered RMSNorm, QK-norm, GELU-tanh/GeGLU; drive attention from
  `Gemma3AttentionLayout` (local/global + sliding window + GQA + dual RoPE). Validate each kernel and
  the single-layer/prefill output against the CPU reference (tolerance documented), then add the WARP
  decode session + KV cache (local layers only need a windowed cache).

- **Heap (product path):** the reference materializes `embed_tokens` as FP32 (~671 MB for 270M) — fine
  for parity, not for the product path. The WARP runtime must be heap-light: ByteBuffer/mmap upload,
  and the tied `embed_tokens`/LM-head must not be uploaded twice. The 256k FP32 LM head dominates
  per-token cost (per the feasibility doc) — candidate for a later optimization slice.

- **Workbench native flag (GEMMA-WARP-11):** wire `-Dgemma.runtime=native-warp` to the native package
  path once tokenizer + WARP exist; keep `external-python-transformers` the default until WARP is
  proven on GPU. The external probe path stays intact (untouched this block).

- **GEMMA-WARP-12 perf — done (measured, no optimization).** Real numbers in
  `docs/gemma3-warp-runtime-performance.md` (`Gemma3WarpPerformanceProbeTest`, gated). On this WARP/CPU
  host: native-warp decode ≈ **0.9–1.15 tok/s** vs external ≈ **8.8–10.7 tok/s** (~10× slower); native
  prefill is token-by-token and scales badly (~25 s for a 20-token prompt); native heap ≈ **1.2 GB**
  (`float[]` reference weights). Correct (`" Paris."`) but **verdict WAIT** — usable experimentally behind
  the flag, not the sensible default until the fused/batched pipeline + heap-light weight load land.
  Bottlenecks + optimization order are in the perf doc.

- **GEMMA-WARP-13a heap-light product weight load — done.** The native product path no longer uses the
  `float[]` reference weights. `Gemma3RuntimePackage.loadWarpWeightsHeapLight()` mmaps the SafeTensors
  payload and decodes each layer projection (q/k/v/o, gate/up/down) and the tied embedding/LM head into
  **direct FP32 ByteBuffers** (off-heap) via `Gemma3WeightBufferView`; norm vectors stay `float[]`
  (small). `Gemma3WarpLayerWeights` now carries the seven projections as `WarpWeightSource`s
  (ByteBuffer-or-float[], decision in `WarpDenseProjection.fromWeightSource`); `Gemma3WarpMlp`/`Gemma3WarpLayer`
  build from sources — the `float[]` test/reference path is numerically unchanged. `Gemma3NativeWarpRuntime`
  uses the heap-light loader (so the tied embedding/LM head go through `Gemma3WarpLmHead.fromFp32ByteBuffer`,
  not duplicated host-side). **Heap before/after load delta = 0 MB** on the real model (vs ~1199 MB for the
  `float[]` reference path in WARP-12); the ~1 GB of weights now lives off-heap. Validated: synthetic
  ByteBuffer-projection logits == float[] logits (bit-for-bit); real heap-light load yields
  ByteBuffer-backed weights and still prefills top-1 9079 (" Paris"); existing WARP suites unchanged.
  The heavy perf probe is now opt-in (`-Dgemma.perf.probe=true`) so the default suite doesn't OOM the WARP
  device with the added heap-light real-model test. Reference path stays for parity/tests; perf
  (fused/batched pipeline) is the remaining WAIT item before native-warp could become default.

- **GEMMA-WORKBENCH-CONVERT-1 UI Convert — done.** Gemma's artifact lifecycle was `Gemma3DownloadLifecycle`
  (download/probe-only: `hasCompiler=false`, `convert` returns failed) → the Workbench Convert button
  stayed disabled, so `-Dgemma.runtime=native-warp` couldn't get its package from the UI. New
  `Gemma3PackageLifecycle` (mirrors `T5PackageLifecycle`): `hasCompiler=true`, targets exactly
  `model_gemma3.wdmlpack` (`Gemma3WdmlPackCompiler.DEFAULT_OUTPUT_NAME` — the file
  `Gemma3NativeWarpRuntime` loads), `inspect` reports raw (config + safetensors) + package state via
  `Gemma3RuntimePackage`, `convert` calls `Gemma3WdmlPackCompiler.compile(dir, model_gemma3.wdmlpack, force)`.
  Wired into both product sites (`DownloadPanel` rows + `lifecycleSupplier`, and
  `DefaultModelArtifactService.createDefault`). The runtime's missing-package message now matches the UI
  ("Open the Download tab, select google/gemma-3-270m-it, then run Convert."). `Gemma3DownloadLifecycle`
  is retained only for its existing unit test. Validated: device-free lifecycle tests (raw-complete →
  Convert, empty → Inspect, corrupt package → Repair, exact-name `existingPackage`) and a CPU
  real-convert test (hard-links the model into a temp dir, converts, asserts the package is
  runtime-loadable and the runtime finds it, lifecycle then READY). No native-runtime path change beyond
  the message; no format change; T5/SmolLM2/Qwen lifecycles untouched.

- **GENERATION-STREAMING-1 streaming switch + Gemma native-warp streaming — done.** A family-neutral
  `GenerationOutputMode` (directml-config) reads `-Ddirectml.generation.output=streaming|buffered` (or
  `-Ddirectml.generation.streaming=true|false`); default and unknown/empty → STREAMING. The Workbench
  Summarizer has a `[x] Streaming output` checkbox initialised from that property and toggleable at
  runtime; the chosen mode flows to every run path. `UiTokenSink(boolean streaming)` appends tokens
  inline only when streaming (buffered suppresses inline; the run method appends the model's full text at
  the end), so Qwen/T5/SmolLM2 keep their live streaming and gain a buffered mode without duplicating
  model logic. Gemma native-warp now streams: `Gemma3NativeWarpRuntime.generateStreaming(prompt, …,
  Consumer<String> onTextDelta)` emits the decoded text per visible token (incremental prefix-decode so
  the deltas concatenate exactly to the final text; the stop token is never streamed); buffered uses the
  no-callback `generate(...)`. **Prompt check:** the native-warp branch already routes through
  `Gemma3PromptStrategy` (same `<start_of_turn>user … model` markers as `Gemma3ChatTemplate`) including
  the user's text — the earlier "Okay, I'm ready" output was the model reacting to a summarize-task
  instruction, not a missing prompt; select template "Keines" (NONE) for a direct answer
  (`Gemma3PromptRoutingTest` locks this).
  - Tests: `GenerationOutputModeTest` (default streaming, output/streaming parsing, precedence);
    `Gemma3PromptRoutingTest` (user text wrapped in chat turn markers); the gated real-model
    `nativeWarpStreamingGeneratesParisPerToken` (deltas `[ Paris, ., \n]`, concat == final text, one delta
    per visible token, stop not streamed); the synthetic `Gemma3WarpGeneratorTest.streamingMatchesVisibleResult`
    already covers the generator streaming contract in the default suite.
  - **Test-suite hygiene (env):** the heavy real-model WARP end-to-end tests are now opt-in via
    `-Dgemma.warp.realModel=true` (they each upload ~1 GB to the WARP software device + a ~671 MB off-heap
    embedding; several in one JVM, or concurrently with a running native-warp Workbench, exhaust the WARP
    device / system RAM). The inference test heap was lowered 4g→2g (weights are heap-light now; the
    largest non-gated heap user is the CPU reference at ~1.1 GB), which also leaves more system RAM for the
    WARP device. Synthetic + device-free GPU tests (incl. the streaming contract) stay in the default
    suite. No native-warp default switch; external-python untouched; no perf/fused/batched work.

- **GEMMA-WARP-13b-1 submit/fence/readback instrumentation — done (measure first).** `WarpSubmissionStats`
  (directml-windows-bindings) — process-wide counters incremented at the three blocking chokepoints:
  `D3D12Bindings.executeAndWait` (every Gemma compute-kernel upload/dispatch/readback-copy goes through it
  → submit + fence wait), `D3D12Bindings.readbackFloatsInternal` (readback), and the `MatMulNBitsKernel`
  matvec single-submit path (one submit + fence + readback). Snapshot/diff measures a region. Pure
  counting, no behaviour change. Device-free `WarpSubmissionStatsTest` (default suite) covers the
  counting; the gated `Gemma3WarpSubmissionStatsTest` (`-Dgemma.warp.realModel=true`) measures the real
  model and asserts top-1 9079 (" Paris") still.
  - **Baseline (real Gemma 3 270M, this host):** per **decode** token **~1140 submits / ~1140 fence
    waits / ~344 readbacks**, ~920 ms → **~1.09 tok/s**; **prefill** ~1135 submits + ~342 readbacks
    *per prompt token* (token-by-token). ~1140 submits/token ≈ 18 layers × ~63, because each Gemma
    compute kernel (RMSNorm/QK-norm/RoPE/scores/softmax/value/GeGLU) uploads its 2–3 buffers as
    *separate* `executeAndWait` submits + a dispatch + a readback (4–6 submits each), whereas the
    projections (`MatMulNBitsKernel`) are already a single combined submit. **This is the batching target.**
  - **Already efficient (single combined submit):** q/k/v/o + gate/up/down projections (`MatMulNBitsKernel`).
    **Still per-call / synchronous:** the seven Gemma compute kernels (each does separate buffer uploads +
    dispatch + readback via `D3D12Bindings`). An existing `DirectMlGpuBatch` seam (fire-and-forget submit +
    deferred fence) is available to reduce these in 13b-2/13b-3.
  - **Forwarding:** the inference test task now passes through the opt-in `-D` toggles
    (`gemma.warp.realModel`, `gemma.perf.probe`, `gemma.testModelDir`, `gemma.runtime`,
    `directml.generation.*`) to the test JVM so the gated tests are runnable on demand; unset → gated
    tests stay disabled and the default suite is green.
  - **Next (not in this slice):** 13b-2 batch/execution-context seam for a Gemma layer; 13b-3 Gemma
    decodeStep keeps norm/rope/scores/softmax/value/geglu GPU-resident across one command list with a
    single readback per token. No math/output change intended — output stays " Paris.".

- **GEMMA-WARP-13b-2 GPU-resident buffer/execution-context seam — done (infra, Gemma-first).**
  `WarpGpuBuffer` (directml-windows-bindings) — a resident FP32 D3D12 buffer (`allocate`/`upload`,
  `gpuAddress()`, `readback()`, `close()`); `WarpExecutionContext` — resident buffer factories +
  `dispatch(kernel, uavs, constants, elementCount)` that records+submits one compute dispatch with **no
  readback** (output stays GPU-resident; one submit+fence, counted by `WarpSubmissionStats`). All seven
  Gemma compute kernels gained an **additive** resident overload
  (`WarpGpuBuffer fn(WarpExecutionContext, WarpGpuBuffer…)`): RmsNorm, QkNorm, RoPE, AttentionScores,
  Softmax, AttentionValue, GeGLU. The existing `float[]` APIs are unchanged (separate code paths;
  reused by all current tests + the still-synchronous decodeStep).
  - Validated (`Gemma3WarpResidentPipelineTest`, synthetic, default suite): resident output == float[]
    output (tol 1e-4) for `RMSNorm→GeGLU` and the `RoPE→Scores→Softmax→Value` chain; **readbacks drop**
    rms→geglu 2→1, attn-chain 4→1 (intermediates stay resident, only the final result is read back).
  - Counting: resident `dispatch` and `WarpGpuBuffer.readback()` go through the same `executeAndWait`/
    `readbackFloatsInternal` chokepoints, so `WarpSubmissionStats` measures them correctly; no
    double-count (the resident path uses `GpuComputeKernel`, not `MatMulNBitsKernel.matvec`).
  - **Still per-call:** each resident `dispatch` is its own submit+fence (this slice removes intermediate
    uploads/readbacks, not the per-dispatch fence). **For 13b-3:** rewire `Gemma3WarpLayer.decodeStep`
    (and the q/k/v/o + gate/up/down projections) onto resident buffers end-to-end with one readback per
    token, and optionally coalesce the per-dispatch fences via `DirectMlGpuBatch` (deferred fence) so a
    decode step approaches a single drain — then compare 13b-1 counters before/after on the real model.
    Output must stay " Paris.". No native-warp default switch / external removal / Qwen-Smol-T5 change /
    batched-prefill / windowed-eviction in this slice.

- **GEMMA-WARP-13b-3a resident decodeStep — done (readbacks down ~9×).** Split per the spec: this is 3a
  (resident decode / fewer readbacks); fence/submit coalescing is the separate 3b. New resident plumbing:
  `MatMulNBitsKernel.matvecResident(WarpGpuBuffer in, WarpGpuBuffer out)` (GPU→GPU copy I/O, one combined
  submit, **no readback**; D3D12 buffers implicit-promote) + `WarpDenseProjection.forwardResident`;
  `Gemma3WarpElementAddKernel` (resident residual add) + a two-buffer resident GeGLU
  (`Gemma3WarpGeGluKernel.apply(ctx, gate, up, inter)`, new `GEGLU2_HLSL`) so the MLP needs no host
  concat; `Gemma3WarpMlp.mlp(ctx, x)`; `Gemma3WarpLmHead.logits(ctx, hidden)`; `Gemma3WarpLayer.decodeStepResident`
  (lazily uploads the 6 norm weights as resident buffers; the whole layer runs resident, the only readback
  is the new token's k/v into the host KV cache); `Gemma3WarpDecodeSession.prefillResident/decodeNextResident`
  (resident embed → layers → final norm → tied LM head, one logits readback). The old `float[]` path is
  unchanged (oracle/fallback).
  - **Measured.** Synthetic: prefill readbacks **382 → 41** (5 prompt tokens), resident logits == float[]
    logits (top-1 identical); resident decodeNext == float[] decodeNext. Real Gemma 3 270M: **readbacks/token
    344 → 37** (~2/layer×18 cache k/v + 1 logits), submits/token **1140 → 834** (intermediate uploads
    removed too), **top-1 9079 (" Paris"), next 236761** — identical output.
  - **Acceptance:** readbacks/token deutlich gesunken (344→37); output identical. Submits/fences still high
    (834/token) — each resident dispatch is its own submit+fence; **that is the next slice (13b-3b):**
    coalesce the per-dispatch fences via `DirectMlGpuBatch`/`executeOrDefer` (deferred fence) and/or record
    a layer's dispatches into one command list, so a decode step approaches a single drain. Streaming stays
    live and buffered stays identical (the session's float[] path, used by the workbench runner, is
    unchanged; the resident path is additive and validated equal). No native-warp default switch; external
    untouched; no Qwen/Smol/T5 change; no batched-prefill / windowed-eviction / .wdmlpack-format change.

- **GEMMA-WARP-13b-3b submit/fence-coalescing — done (fences down ~9×, submits ~halved).** Two coalescing
  mechanisms layered on the 13b-3a resident path, both kept behind an active `DirectMlGpuBatch` so the old
  synchronous resident/`float[]` paths remain the always-correct fallback:
  1. **Deferred fences.** `WarpExecutionContext.dispatch` now submits via `D3D12Bindings.executeOrDefer`
     (fire-and-forget, fence coalesced into the batch drain) and releases its command list/allocator —
     fixing a pre-existing one-list/allocator leak per dispatch. `MatMulNBitsKernel.matvecResident` grows a
     deferred branch (`matvecResidentDeferred`, fresh command list so the batch can `AddRef`/retain it;
     recording extracted to `recordMatvecResident`) used when a batch is active. `Gemma3WarpLayer.decodeStepResident`
     opens a **per-layer** batch (bounds retained command lists on the memory-sensitive WARP device) and
     drains once at layer end; `Gemma3WarpDecodeSession.residentLogits` batches the final-norm + LM head.
     Because dispatches are deferred, intermediate buffers must outlive the drain — the layer collects them
     in a scratch list closed only after the batch closes (an eager mid-layer `close()` was freeing GPU
     memory a not-yet-executed command list still referenced → `DEVICE_REMOVED`); `Gemma3WarpMlp.mlp(ctx, x,
     scratch)` registers its gate/up/activated the same way.
  2. **No per-output zero-init.** Every resident kernel fully overwrites its output (RMSNorm, QK-norm, RoPE,
     scores incl. the masked `-1e30f` sentinel, softmax, value, GeGLU, element-add, matvec), so
     `WarpExecutionContext.allocate` now uses `WarpGpuBuffer.allocateUninitialized` — dropping the
     synchronous zero-init upload (one submit **and** one fence) per kernel output, which was the dominant
     remaining sync cost.
     Instrumentation: `WarpSubmissionStats.recordSubmit()` (deferred submit, no fence) + `recordFenceWait()`
     (the batch drain) so the counters stay honest.
  - **Measured.** Synthetic prefill (5 tokens): resident **submits 955 → 533, fenceWaits 554 → 132** (float
    path 1269/1269), logits == float[] (top-1 identical). Real Gemma 3 270M decode/token:
    **fenceWaits 834 → 93** (~9×; readback drains + key/value uploads + per-layer drains), **submits 834 → 454**
    (zero-init uploads removed), **readbacks 37** (unchanged), **top-1 9079 (" Paris"), next 236761** — identical
    output.
  - **Acceptance:** output identical, " Paris" preserved, readbacks stay low (37), submits/fences/token both
    deutlich below ~834 (454 / 93). No native-warp default switch; external-python untouched; workbench
    runner still uses the stable `float[]` session path (resident/batched is additive + validated equal).
  - **Still per-dispatch (not coalesced into one list):** each deferred dispatch is still its own
    `ExecuteCommandLists`, so submits track op count; the remaining fences are the unavoidable k/v readbacks,
    the growing key/value uploads, and one drain per layer. Further submit reduction would mean recording a
    whole layer's dispatches into a single command list (and/or deferring the key/value uploads with batch-
    retained upload buffers) — a larger, riskier change deferred to a later slice.

- **GEMMA-WORKBENCH-PROFILING-1 profile output + runtime UI — done.** Two Workbench usability changes, no
  runtime-path change (the native generator still uses the synchronous `float[]` `decodeStep` path via
  `Gemma3WarpGenerator`, so the profile reflects that path; switching it to the 13b-3b resident path to
  surface those gains in the Workbench is a separate future perf slice).
  - **Runtime mode in the UI (Ziel 3).** `SummarizerPanel` gains a "Gemma runtime:" selector (External
    Python / Transformers · Native Java/WARP (experimental)), enabled only for Gemma. The choice lives in
    `WorkbenchModel.gemmaRuntimeMode` and is authoritative; `runGemma3Generation` branches on it instead of
    `-Dgemma.runtime`. The legacy `-Dgemma.runtime=native-warp` flag remains only as a seed for the initial
    selection (compat alias), so no JVM flag is needed for normal use. External stays the default.
  - **Profiling (Ziele 1/5).** `Gemma3NativeWarpRuntime` measures the load phases (package open, tokenizer
    load, weight load, WARP/session init) and generation phases (tokenize, prefill = time to first token,
    decode total + avg/token, detokenize) and captures the `WarpSubmissionStats` deltas over the generate
    region (submits / fence waits / readbacks, + per-token), returned as a `Gemma3NativeWarpProfile` on the
    `Result`. `Gemma3NativeWarpProfileReport` (Swing-free, unit-tested) formats the detailed block (incl.
    runtime mode / backend / output mode / package / tokenizer / prompt template / effective prompt chars /
    token counts) or a short summary.
  - **Toggle (Ziel 2).** A "Show runtime profile" checkbox (initial value from
    `-Ddirectml.generation.profile`, checkbox authoritative; default off) selects detailed vs summary.
  - **Messages (Ziel 4).** Missing package → the existing actionable "compiled model_gemma3.wdmlpack …
    Download tab → Convert"; runtime mode lines use `Gemma3RuntimeMode.displayLabel()`
    (`external-python-transformers` / `native-warp-experimental`). Streaming stays the default; buffered
    stays available. No native-warp default; external-python kept; no fused pipeline / batched prefill /
    windowed-eviction / downloader / .wdmlpack-format change; Qwen/T5/SmolLM2 untouched. Full
    `:directml-inference` + `:directml-encoder` + `:directml-workbench` regression green.

- **GEMMA-WARP-13b-4 resident/batched as native-warp product path — done.** The native-warp Workbench path
  now drives the validated GPU-resident/batched decode (13b-3a/3b) instead of the synchronous `float[]`
  `decodeStep`, so the Workbench actually gets the 13b-3b gains and the profile shows the coalesced counters.
  - **Switch.** `Gemma3WarpExecutionMode` (SYNC | RESIDENT, `-Dgemma.warp.execution`, default RESIDENT).
    `Gemma3WarpGenerator` gains an execution-mode ctor and picks `session.prefillResident/decodeNextResident`
    vs `prefill/decodeNext` accordingly (existing ctors stay SYNC, so `Gemma3WarpGeneratorTest` keeps
    validating the float[] path). `Gemma3NativeWarpRuntime` resolves the mode (default RESIDENT) and passes
    it; the resolved mode is carried in the profile and shown in the report ("execution: resident-batched").
  - **Fallback.** `-Dgemma.warp.execution=sync` restores the float[] decode path (debug/fallback); no global
    default change elsewhere, external-python untouched.
  - **Per-token counters fixed to decode steady-state.** The profile now snapshots WARP counters at the
    first-token boundary so the per-token figures cover only the decode region (not prefill-amortised); the
    totals stay whole-generate.
  - **Measured.** Synthetic: resident generate == sync generate (identical ids). Real Gemma 3 270M
    (`-Dgemma.warp.realModel=true`): native-warp now resident; decode steady-state **454 submits / 93 fence
    waits / 37 readbacks per token** (vs the float[] path's ~834 fenceWaits + ~344 readbacks), generate
    totals 3729 / 846 / 291 over a 6-token prompt + 3 outputs, decode avg ≈ 426 ms/token, **top-1 " Paris"**
    unchanged. Streaming stays live; buffered yields the same text.
  - Tests: `Gemma3WarpExecutionModeTest` (default resident, sync override, parsing); `Gemma3WarpGeneratorTest`
    resident==sync equivalence; `Gemma3NativeWarpRuntimeTest` gated real asserts resident default + " Paris" +
    decode per-token bounds; `Gemma3NativeWarpProfileReportTest` (decode-region per-token, execution line).
    No native-warp global default; no fused pipeline / batched prefill / windowed-eviction / downloader /
    .wdmlpack-format change; Qwen/T5/SmolLM2 untouched. Full `:directml-inference` + `:directml-encoder` +
    `:directml-workbench` regression green.

- **WORKBENCH-CLEANUP-STREAMING-ONLY-1 remove extra UI controls — done.** Reverts the two extra visible
  controls added by GEMMA-WORKBENCH-PROFILING-1 while keeping the profiling infrastructure as a debug path.
  - **Removed (UI + model state):** the "Gemma runtime:" combo + label and its `WorkbenchModel.gemmaRuntimeMode`
    state/getter/setter (and the `-Dgemma.runtime` seed); the visible "Show runtime profile" checkbox + its
    state. The only new visible control that remains is **Streaming output** (default on).
  - **Gemma runtime decision** now comes from the general **Backend**: `SummarizerPanel.gemmaUsesNativeWarp(backend)`
    = `backend == WARP` → native Java/WARP path (Backend≠WARP → external Python probe). No Gemma-specific
    selector; `-Dgemma.runtime` no longer participates in product/UI logic (kept only as a label enum +
    deprecated no-op).
  - **Profiling kept as debug path:** `Gemma3NativeWarpProfile`/`Report`, `WarpSubmissionStats`, the runtime
    measurement and `Gemma3NativeWarpRuntimeTest`/`Gemma3NativeWarpProfileReportTest` are unchanged. The
    detailed profile block is emitted only when started with `-Ddirectml.generation.profile=true`
    (`SummarizerPanel.SHOW_PROFILE`); otherwise the short summary prints. No checkbox.
  - Tests: `GemmaRuntimeSelectionTest` rewritten — asserts no Gemma runtime selector field, no visible
    profile checkbox field, no `WorkbenchModel.get/setGemmaRuntimeMode`, `gemmaUsesNativeWarp` is backend-based
    and ignores `-Dgemma.runtime`, streaming is the default. Missing-package Convert hint unchanged. No
    profiling-infrastructure deletion; `WarpSubmissionStats` kept; streaming control kept; no perf/downloader/
    .wdmlpack-format/Qwen-Smol-T5 change. Full `:directml-inference` + `:directml-encoder` +
    `:directml-workbench` regression green.

- **GEMMA-WARP-13c GPU-resident KV cache — done.** Removes the last per-layer host round-trip from the
  native-warp resident path: previously `decodeStepResident` read the new token's k/v back to the CPU
  (`kr.readback()`/`v.readback()`), wrote a host `Gemma3WarpKvCache`, and re-uploaded the whole visible
  K/V each layer (~2 readbacks + 2 uploads/layer → readbacks/token ≈ 38). Now k/v stay on the GPU.
  - **New:** `Gemma3WarpResidentKvCache` + `Gemma3WarpResidentKvLayerCache` (per-layer device K/V buffers,
    capacity×kvDim, full history; window applied as a read-time mask, no eviction). `Gemma3WarpKvAppendKernel`
    appends the new token's k/v into the cache via a **UAV write** (`cache[pos*kvDim + i] = src[i]`), so the
    `executeOrDefer` UAV barrier orders it before the attention read — same resident model as the other
    kernels, batch-safe. `WarpExecutionContext.copyRegionInto` does the rare cache growth (doubling + prefix
    copy) synchronously outside the batch. The attention-scores/value kernels now accept a cache buffer
    larger than the visible region (validation `>=` instead of `==`).
  - **decodeStepResident:** appends kr/v into `cache.layer(l)` and runs scores/value reading the cache K/V
    buffers in place; no `kr.readback()`/`v.readback()`, no per-layer upload. The cache buffers are
    persistent (not tracked in the layer scratch). The session grows the cache before the per-layer batch.
    The host `Gemma3WarpKvCache` stays only for the sync/`float[]` debug path.
  - **Measured.** Synthetic prefill (5 tokens): resident readbacks **41 → 1**, fenceWaits **132 → 52**,
    submits 533 → 493 (vs float path 1269/1269); resident logits == float[] (top-1 identical). Growth+window
    parity: GPU-KV ids == float[] ids across 45 positions (initial cap 32 → grows) with slidingWindow=2
    (local windowed + full layers), 1 readback per logits call. Real Gemma 3 270M decode/token:
    **readbacks/token 38.42 → 1.00**, **fence waits/token 96.58 → 21.00**, submits/token 471 → 418,
    decode avg ≈ 442 ms/token, **top-1 9079 (" Paris"), next 236761** — identical output. Streaming live;
    buffered same text.
  - Tests: `Gemma3WarpResidentDecodeStepTest` (resident==float[] now via GPU-KV; readbacks ≤ 3;
    growth+window parity); `Gemma3WarpGeneratorTest` resident==sync; gated real " Paris" with readbacks/token
    ≤ 3. No second product path (sync float[] kept as debug/fallback via `-Dgemma.warp.execution=sync`); no
    windowed eviction / batched prefill / UI control / downloader / .wdmlpack-format change; Qwen/Smol/T5
    untouched. Full `:directml-inference` + `:directml-encoder` + `:directml-workbench` regression green.

- **GEMMA-WARP-13d command-list coalescing — done.** After 13c the readbacks were ~1/token but submits
  stayed ~418/token (one ExecuteCommandLists per kernel), and on WARP the per-submit driver overhead is the
  decode bottleneck. This slice records consecutive UAV dispatches into one command list.
  - **Seam:** `WarpExecutionContext` gains a coalescing scope (`beginRecording`/`flushRecording`). Inside
    it, `dispatch()` accumulates UAV kernels (norms, QK-norm, RoPE, KV append, attention, GeGLU,
    element-adds) into one lazily-opened command list with a `uavBarrier` after each, submitted once via
    `executeOrDefer` (deferred into the active per-layer `DirectMlGpuBatch`). `matvec()` (new; used by
    `WarpDenseProjection.forwardResident`) flushes the pending list, then runs the kernel's **standalone**
    resident matvec — its copy-based I/O can't safely share a UAV list, so the math is byte-identical to the
    non-coalesced path (the spec's "kein mathematisches Risiko"). `Gemma3WarpLmHead.logitsResident` returns
    the resident logits buffer so the final norm + LM-head coalesce before the one readback.
  - **Fences kept low:** the per-layer `DirectMlGpuBatch` (13c) is retained, so all coalesced lists + matvecs
    are deferred and the batch drains once per layer — fenceWaits/token stay ~21 (not worse). Lifetime
    unchanged: scratch buffers freed only after the batch drains.
  - **Measured.** Synthetic prefill (5 tokens): resident submits 533 → 273, fenceWaits 132 → 52, readbacks 1;
    logits == float[] (top-1 identical); growth+window parity holds. Real Gemma 3 270M decode/token:
    **submits/token 418 → 220**, **fence waits/token 21 (unchanged)**, **readbacks/token 1**, **decode avg
    ~442 → ~294 ms/token** (~33 % faster), top-1 9079 (" Paris"), next 236761 — identical output. Streaming
    live; buffered same text.
  - Tests: `Gemma3WarpResidentDecodeStepTest` (resident == float[]; `syntheticResidentPrefillCoalescesCommandLists`
    asserts resident submits < float-path submits; gated real submits/token < 300); `Gemma3WarpGeneratorTest`
    resident == sync. No mathematical change (matvec standalone); no UI control / downloader / .wdmlpack-format
    change; Qwen/Smol/T5 untouched. Full `:directml-inference` + `:directml-encoder` + `:directml-workbench`
    regression green.
  - **Still open (toward batched prefill / fewer submits):** the ~220 submits/token are dominated by the 7
    standalone DML-GEMM matvec projections/layer (kept standalone for exact numerics) + ~5 coalesced UAV runs/
    layer. Pushing toward layer-count (~20) needs the matvec inside the coalesced list — either a UAV matvec
    shader (a tiny numeric change) or in-list resource-state transitions for the DML copies. **Batched prefill**
    (processing all prompt tokens in one pass instead of token-by-token) is the separate next lever for prefill
    latency and is still open.

- **GEMMA-WARP-13e batched prefill — done.** Prefill no longer repeats the per-token decode step for every
  prompt token; it processes the whole prompt in one pass per layer.
  - **New:** `MatMulNBitsKernel.matmulBatchResident` (resident batched matmul `out[M,N]=in[M,K]·Wᵀ` on the
    shared batch shader, one submit, GPU→GPU I/O); `WarpDenseProjection.forwardResidentBatched` +
    `supportsBatchedResident`; `WarpExecutionContext.matvecBatched`; `Gemma3WarpRowCopyKernel` (UAV
    gather/scatter `dst[dstOff+i]=src[srcOff+i]`, coalesces with the other dispatches);
    `Gemma3WarpMlp.mlpBatched`; `Gemma3WarpLayer.forwardPrefillBatched` (batched q/k/v/o + MLP projections,
    per-position QK-norm/RoPE/attention reusing the validated single-query kernels, all k/v appended to the
    GPU-resident KV cache); `Gemma3WarpDecodeSession.prefillResidentBatched` (batched embedding → layers →
    final norm + LM head once for the last position).
  - **Product path:** `Gemma3WarpGenerator` resident prefill now calls `prefillResidentBatched`; the
    token-by-token `prefillResident` stays as the debug/fallback oracle (the `float[]` sync path is also
    unchanged). Long prompts (> the batch cap) fall back to per-position projections automatically.
  - **Exact-ish parity:** the batched matmul uses the shared tiled batch shader vs the single matvec's DML
    GEMM, so results match within fp32 tolerance (logits assertClose + identical top-1), not bit-for-bit.
  - **Measured.** Synthetic prefill (7 tokens, slidingWindow=2): batched == tokenwise (logits + top-1 +
    decodeNext-after), submits 374 → 66. Real Gemma 3 270M (6-token prompt): prefill submits 1417 → 261
    (~5.4×), readbacks 1, fenceWaits 228 → 260 (synchronous batched matmuls), top-1 9079 (" Paris") for both
    tokenwise and batched; decodeNext after batched prefill works; streaming " Paris" live. Decode unchanged
    (220 submits/token, 294 ms/token from 13d).
  - Tests: `syntheticBatchedPrefillMatchesTokenwiseResident` (parity + decodeNext-after + submits<tokenwise);
    gated real asserts batched top-1 " Paris" + prefill submits < tokenwise; `Gemma3WarpGeneratorTest` /
    streaming unchanged. No UI / downloader / .wdmlpack-format change; no windowed eviction; Qwen/Smol/T5
    untouched. Full `:directml-inference` + `:directml-encoder` + `:directml-workbench` regression green.
  - **Still open:** prefill fence waits rose slightly (synchronous batched matmuls) — deferring them (shared
    scratch makes that non-trivial) and coalescing the matvec projections into the layer list (a UAV matvec
    shader or in-list transitions) remain the levers toward ~layer-count submits. The batched matmul reuses
    the shared static batch scratch, so prefill batches run one matmul at a time (fine; bounded VRAM).

- **GEMMA-WARP-13f warm/cold profiling + shader warm-up — done.** Measurement infrastructure only (no
  kernel/model change), so cold one-time costs no longer hide in the numbers.
  - **New:** `Gemma3WarpWarmup` (runs one short batched prefill + a few resident decode steps to trigger all
    lazy shader compiles; changes no logic) and the gated `Gemma3WarpSteadyStateProfileTest` (cold prefill/
    decode → warm-up → warm prefill+16-token decode for 6/25/100-token prompts, prefill and decode counters
    measured separately; asserts " Paris" smoke, warm-up doesn't change token IDs, counters present — no ms
    assertions).
  - **Measured (real Gemma 3 270M, this host).** COLD prefill(6 tok) 2018 ms / 373 submits, first decode
    316 ms; lazy shader compile ≈ the cold−warm prefill gap (~480 ms). WARM prefill **261 / 280 / 355
    submits** for **6 / 25 / 100**-token prompts (≈ constant in prompt length — the 13e batched win) at
    1538 / 5473 / 20350 ms; WARM decode **~250 ms/token** over 16 tokens, **220 submits/token, 21 fence
    waits/token, 1 readback/token** (stable, matches 13c/13d). " Paris" green cold and warm.
  - **Findings → next optimization.** (A) **Decode is submit-bound:** ~220 submits/token ≈ ~250 ms/token
    (~1.1 ms/submit on WARP) — coalescing the 7 standalone DML-GEMM matvec projections/layer into the layer
    command list (a UAV matvec shader or in-list resource-state transitions) is the highest-leverage next
    step and should cut decode time substantially. (B) **Long prefill is now compute-bound, not submit-bound:**
    submits are ~constant (~300) but time grows with prompt length (per-position attention loop is O(seqLen)
    dispatches + larger matmuls) — multi-query batched attention would help long prompts but is a bigger/
    riskier kernel slice. **Recommended next: (A).**
  - Full `:directml-inference` + `:directml-encoder` + `:directml-workbench` regression green (the profile
    test is gated/opt-in).

- **GEMMA-WARP-13g projection/matvec submit reduction — done (DML numerics preserved).** The 7 standalone
  DML-GEMM matvec projections/layer are now recorded INTO the layer's coalesced command list instead of one
  submission each.
  - **How (no new matmul, no numeric change):** `MatMulNBitsKernel.recordResidentInto(cl, in, out, arena)`
    records the existing resident matvec (the DML/INT4/FP32 dispatch + its staging copies, unchanged) into a
    caller-owned command list. The kernel's private inputBuf/outputBuf go COMMON→…→COMMON (self-contained,
    as in the standalone path); only the external `in`/`out` are bracketed with resource-state transitions
    the shared list needs (`in` UAV↔COPY_SOURCE around the input copy; `out` COPY_DEST→UAV after the output
    copy) — valid because matvec inputs/outputs always rest in UNORDERED_ACCESS in the recording.
    `WarpExecutionContext.matvec` now records into the open list (no flush, no separate submit) inside a
    coalescing scope. So a whole decode layer = one command list = one submit. The standalone matvec stays
    as the non-coalesced fallback; the batched-prefill matmul (`matvecBatched`) is unchanged.
  - **Measured.** Synthetic resident prefill submits 273 → 52 (logits == float[] oracle; growth+window
    parity holds). Real Gemma 3 270M decode/token: **submits 220 → 21** (~layer count), **fence waits 21
    (unchanged)**, **readbacks 1 (unchanged)**, **top-1 9079 (" Paris"), next 236761 — identical** (DML
    matvec untouched). Streaming live; buffered same text.
  - **Key finding — decode is compute-bound, not submit-bound.** Cutting submits 10× (220→21) left warm
    decode at **~221–252 ms/token** (essentially unchanged from 13f). So on the WARP software rasterizer the
    bottleneck is the matvec FLOPs themselves (the FP32/DML GEMM on a CPU rasterizer), not the per-submit
    driver overhead. The earlier 13f "submit-bound" guess was wrong.
  - **Next optimization → matvec COMPUTE, not submits.** Options: a better-parallelised WARP FP32 matvec
    (the current shader is ~one thread per output row; tiling/vectorising the 640×{1024,2048} GEMVs would
    help), INT4 weights to cut memory bandwidth, or accepting WARP's CPU-rasterizer ceiling and validating
    on a hardware GPU. Long-prefill compute (per-position attention, O(seqLen)) is the other lever. Submits
    and readbacks are now effectively solved.
  - Tests: `Gemma3WarpResidentDecodeStepTest` (resident == float[]; gated real decode submits/token < 100 +
    " Paris"); `syntheticBatchedPrefillMatchesTokenwiseResident` re-tuned to a 64-token prompt (batched <
    tokenwise once long enough; the short-prompt inequality no longer holds because tokenwise now also
    coalesces matvecs). No UI / downloader / .wdmlpack-format change; no windowed eviction; Qwen/Smol/T5
    untouched. Full `:directml-inference` + `:directml-encoder` + `:directml-workbench` regression green.

- **GEMMA-WARP-14a matvec compute benchmark + decision — done (measurement, no production change).** After
  13g found decode compute-bound, this benchmarks the three matvec implementations for the real Gemma 270M
  projection shapes (M=1 decode) and picks a direction. `Gemma3WarpMatvecBenchmarkTest` (gated opt-in,
  `-Dgemma.warp.bench=true`) builds each impl on a fresh device (DML on `directml`, the custom shaders on
  `warp`), warms up, and times `matvecResident`. (`directml-inference/build.gradle` now also forwards
  `gemma.warp.bench` / `gemma.warp.execution` / `directml.generation.profile`.)
  - **Measured (this host, ms/call, M=1):** DML-GEMM is fastest for **every** shape:

    | shape | N | K | DML-GEMM | WARP-FP32 | INT4-WARP |
    |-------|---|---|----------|-----------|-----------|
    | q_proj | 1024 | 640 | 0.194 | 0.633 | 0.544 |
    | k/v_proj | 256 | 640 | ~0.15 | ~0.33 | ~0.31 |
    | o_proj | 640 | 1024 | 0.145 | 0.572 | 0.590 |
    | gate/up_proj | 2048 | 640 | ~0.149 | ~0.79 | ~0.88 |
    | down_proj | 640 | 2048 | 0.149 | 0.873 | 0.953 |
    | lm_head | 262144 | 640 | **1.034** | **51.835** | **64.376** |

    Estimated matvec ms/decode-token (18 layers × 7 proj + lm_head): **DML-GEMM ≈ 20.5, WARP-FP32 ≈ 129.6,
    INT4-WARP ≈ 145.0**. The custom one-thread-per-row WARP shaders are 3–6× slower than DML for the small
    projections and ~50–60× slower for the 256k-row LM head.
  - **DECISION: keep DML-GEMM** for the resident matvec — it is already the product path and is decisively
    fastest here. Do **not** switch to the custom FP32/INT4 WARP shaders; INT4 also wouldn't help (and would
    need quantised Gemma weights).
  - **Bigger finding — matvec is NOT the decode bottleneck.** DML matvec is only **~20 ms** of the ~250 ms
    decode/token. The remaining ~230 ms is the many small per-layer element/attention dispatches (RMSNorm,
    QK-norm, RoPE, KV-append, scores, softmax, value, GeGLU, element-adds — ~15/layer × 18 ≈ 270 tiny
    dispatches) plus a UAV barrier between each, whose fixed per-dispatch/per-barrier cost on the WARP
    software rasterizer dominates. So 13g's "compute-bound on matvec" was imprecise: it is dispatch/
    barrier-bound on the small element kernels. **Next lever: kernel fusion** (fuse RMSNorm+residual, the
    QK-norm/RoPE chain, the attention chain) to cut dispatch + barrier count — not a faster matvec. A
    hardware GPU (vs the WARP CPU rasterizer) would also reduce the per-dispatch overhead.
  - No production/numeric change; gated benchmark opt-in; Qwen/Smol/T5 untouched. Full `:directml-inference`
    + `:directml-encoder` + `:directml-workbench` regression green.

- **GEMMA-WARP-14b fuse attention (scores+softmax+value) — done.** Replaces the staged 3-dispatch attention
  middle with one fused kernel, and adds dispatch/barrier counters so the per-token cost is visible now that
  many dispatches share one submit.
  - **Counters (Part 1):** `WarpSubmissionStats` gains `dispatches` + `uavBarriers` (bumped in
    `GpuComputeKernel.recordDispatch`, the DML-GEMM dispatch, and `D3D12Bindings.uavBarrier`); `Snapshot`
    carries them; `Gemma3NativeWarpProfile`/`Report` expose `dispatches/token` + `uav barriers/token` (decode
    region) in the `-Ddirectml.generation.profile` block.
  - **Fused kernel (Part 2):** `Gemma3WarpFusedAttentionContextKernel` computes
    `context = softmax(scale·q·K_visible)·V_visible` in one dispatch — flash-attention online softmax (running
    max/sum + rescaled accumulator, no materialised `[seqLen]` scores), one thread group per head, thread c
    owns output dim c, per-key dot via a group reduction. Same Gemma layout: GQA (`kvHead=head/groupsPerKv`),
    causal + sliding-window via `[firstValid, queryPos]`, explicit head_dim (≤256). `Gemma3WarpLayer`
    (`attentionContext` helper) uses it in both resident decode and batched prefill; `-Dgemma.warp.attention=staged`
    forces the old staged scores→softmax→value path (kept as the parity oracle/fallback); default = fused.
  - **Numerics:** fused vs staged differ only by fp32 accumulation order; parity tol abs 1e-3 + rel 1e-3,
    identical top-1/token-IDs. Validated: kernel-level fused==staged (GQA 4/1, head_dim=256, full + local +
    single-position + no-GQA + mid-head_dim); resident decode fused==staged token IDs (fed identical tokens).
  - **Measured (real Gemma 3 270M, decode/token):** dispatches **416 → 380**, uav barriers **435 → 399**
    (−36 each = the 2 saved attention dispatches/barriers × 18 layers); submits/fence waits/readbacks
    **21/21/1 unchanged**; top-1 9079 (" Paris") identical; streaming live, buffered same text. Warm decode
    avg/token is noisy/flat (~210–280 ms): fused trades 3 light dispatches for 1 heavier (the per-key group
    reduction serialises), so the saved per-dispatch overhead is roughly offset on the WARP CPU rasterizer —
    a small win at long context (100-tok: 208 vs 233 ms), ~equal at short. `build.gradle` forwards
    `gemma.warp.attention`.
  - **Next fusion candidate:** the residual+RMSNorm pairs (post-attention add → pre-FF norm; post-FF add →
    next input norm) and the QK-norm+RoPE chain are the remaining many-small-dispatch clusters; or move off
    the WARP software rasterizer to a hardware GPU (the per-dispatch overhead is the WARP ceiling). No UI
    change; gated/opt-in real + profile tests; Qwen/Smol/T5 untouched. Full `:directml-inference` +
    `:directml-encoder` + `:directml-workbench` regression green.
