# T5 real-model correctness cert (prep + CERT-1..4 + closeout)

> **Status (T5-PRODUCT-CLOSEOUT): all four curated T5 models are real-certified** — `google-t5/t5-small`,
> `google/flan-t5-small`, `Salesforce/codet5-small`, `Salesforce/codet5-base-multi-sum` (CPU reference == WARP
> mixed, greedy; token ids + text identical; T5-REALMODEL-CERT-1..4). **Certification scope:** exactly these four;
> any other or newly-added T5/CodeT5 model must run its own gated cert before being called certified — there is no
> blanket guarantee for arbitrary T5-family models. CPU stays the always-validated path; WARP/AUTO are the mixed
> DirectML path, now certified against CPU for the four. A prebuilt `.wdmlpack` is mandatory and Download → Convert
> is the prerequisite. No Python on any T5 path. Real-model + synthetic certs stay opt-in; downloaded `model/...`
> artifacts are git-ignored.

`T5-CORRECTNESS-CERT-1` could only measure the WARP/AUTO mixed path against the CPU reference on a *synthetic*
package (verdict **D** — no real T5 artifact present). This note prepares the **primary** real-model so the gated
`T5MixedRuntimeCorrectnessCertTest` real-model cert can run end-to-end.

**Primary cert model: `google-t5/t5-small`** — smallest, plain T5 (publishes `model.safetensors` directly, no
PyTorch-checkpoint step), fewest tokenizer special cases.

## Verdict: B — preparable with the existing Download/Convert path; no code change needed

The full chain is already wired; nothing in the runtime/downloader/convert is missing:
- **Download** is offered in the Workbench Download tab: *"Download google-t5/t5-small (SafeTensors smoke-test)"*
  (`DownloadPanel.addT5Row` → `ModelDownloadUrls.manifestForGoogleT5Small`). Files come from the public, ungated
  repo `https://huggingface.co/google-t5/t5-small/resolve/main/<file>`.
- **Convert** is the same row's package action: `ModelArtifactRow(ModelFamily.T5, …, T5PackageLifecycle::new)` →
  `T5PackageLifecycle.convert` compiles `model_t5.wdmlpack` from `config.json` + `model.safetensors`.
- **Target dir**: `<modelRoot>/t5-small` (default `<repo>/model/t5-small`).

(The only reason `T5-CORRECTNESS-CERT-1` returned D was that no artifact had been downloaded yet — not a missing
building block.)

## Expected local artifacts (in `<repo>/model/t5-small`)

| File | Required | Used for |
|------|----------|----------|
| `model.safetensors` | yes | convert source (weights) |
| `config.json` | yes | convert source + engine config |
| `tokenizer.json` | yes | inference tokenizer (`JsonT5Tokenizer`) |
| `tokenizer_config.json` | yes (download) | tokenizer metadata |
| `spiece.model` | yes (download) | SentencePiece source |
| `generation_config.json` | optional | generation defaults |
| `model_t5.wdmlpack` | produced by Convert | the runtime package the cert loads |

The engine accepts either a `tokenizer.json` (google-t5) **or** `vocab.json`+`merges.txt` (CodeT5); t5-small uses
`tokenizer.json`. Convert needs only `config.json` + `model.safetensors`; the tokenizer is validated separately at
inference time.

## How to prepare

**Option 1 — Workbench (recommended):** open the Download tab → click *"Download google-t5/t5-small
(SafeTensors smoke-test)"* → after it completes click the row's Convert/package action to produce
`model_t5.wdmlpack`.

**Option 2 — manual:** download the six files above from
`https://huggingface.co/google-t5/t5-small/resolve/main/<file>` into `<repo>/model/t5-small`, then convert via the
Workbench Download tab (Convert) or `T5WdmlPackCompileTool` (the same `T5WdmlPackCompiler` the lifecycle uses).

## Running the real-model cert

Once `<repo>/model/t5-small/` contains the artifacts above **and** `model_t5.wdmlpack`, the cert auto-resolves
that directory (no override needed):

```bash
JAVA_HOME="C:/Program Files/Zulu/zulu-21" GRADLE_USER_HOME=.chatgpt/gradle-home \
  bash ./gradlew --no-daemon --offline :directml-inference:test \
  --tests '*T5MixedRuntimeCorrectnessCertTest' \
  -Dt5.correctness.cert=true \
  -Dt5.realModel=true
```

With the model in a non-default directory, point the cert at it. The primary model also honors a clean generic
override (it applies to `google-t5/t5-small` only):

```bash
  -Dt5.testModelDir="D:/models/t5-small"
# or the verbatim per-model key (any T5 model):
  -Dt5.testModelDir.google-t5/t5-small="D:/models/t5-small"
```

`-Dt5.realModel=true` alone keeps the synthetic device cert off; add `-Dt5.correctness.cert=true` to run both. The
real-model cert compares the `reference` vs `warp` runs (token preview + text) for each present model and skips any
model whose directory is absent; it fails only if **no** real model is present while `-Dt5.realModel=true` is set.

These toggles are forwarded to the test JVM by `directml-inference/build.gradle`
(`t5.correctness.cert`, `t5.realModel`, `t5.testModelDir`, and any `t5.testModelDir.<modelId>`).

## Flan-T5 / CodeT5 (optional, analogous)

The same path works for the other three curated models once their artifacts are present, and the cert already
includes them (each skips cleanly until downloaded):
- `google/flan-t5-small` → `model/flan-t5-small` (SafeTensors; manifest `manifestForGoogleFlanT5Small`).
- `Salesforce/codet5-small` and `Salesforce/codet5-base-multi-sum` → `model/codet5-small` /
  `model/codet5-base-multi-sum` (these publish `pytorch_model.bin` + `vocab.json`/`merges.txt`; the convert uses the
  restricted Torch state-dict import path). t5-small is the primary because it avoids the Torch-checkpoint and
  BPE-tokenizer special cases.

## Status

`google-t5/t5-small` is **real-certified (T5-REALMODEL-CERT-1, verdict A)**: token ids
`[644:▁Das, 4598:▁Haus, 229:▁ist, 19250:▁wunderbar, 5:.]` / text `Das Haus ist wunderbar.` identical on CPU and WARP.

`google/flan-t5-small` is **real-certified (T5-REALMODEL-CERT-2, verdict A)**: prompt `"Answer the question: what is
the capital of France?"` → token ids `[1410:▁France]` / text `France` identical on CPU and WARP (greedy;
executionMode `warp-encoder-boundary+warp-decoder-boundary+warp-lm-head`).

`Salesforce/codet5-small` is **real-certified (T5-REALMODEL-CERT-3, verdict A)** — the CodeT5 special case
(`pytorch_model.bin` Torch checkpoint + `vocab.json`/`merges.txt` BPE tokenizer, no safetensors/SentencePiece). Prompt
`"def add(a, b):\n    return a + b"` → token ids `[1:<s>, 32099:<extra_id_0>, 30::]` identical on CPU and WARP. It is
a pretrained base checkpoint, so the output is span-corruption sentinels (low quality) but byte-identical across
paths — the cert measures CPU-vs-WARP parity, not output quality.

`Salesforce/codet5-base-multi-sum` is **real-certified (T5-REALMODEL-CERT-4, verdict A)** — the largest curated T5
(base, ~990 MB, 260 tensors; Torch checkpoint + BPE). Prompt `"def add(a, b):\n    return a + b"` → token ids
`[1:<s>, 3495:Sum, 21872:marize, 2795:Ġtwo, 6548:Ġterms, 263:Ġ.]` / text `Summarize two terms.` identical on CPU and
WARP (this model is fine-tuned for summarization, so the output is coherent).

**The entire curated T5 family is now real-certified.** The cert asserts parity for any present model. The
`model/t5-small/`, `model/flan-t5-small/`, `model/codet5-small/` and `model/codet5-base-multi-sum/` artifacts are
git-ignored. No Gemma/Qwen/SmolLM2/Phi change. See `workbench-model-status.md` for the full result.
