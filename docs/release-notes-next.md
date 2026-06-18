# Release notes (next) — draft

Working draft for the next release. **No tag and no publication yet.** The published Maven artifacts remain the
embeddings/reranking library (`directml-config`, `directml-windows-bindings`, `directml-encoder`,
`directml-runtime`); everything under "Workbench text generation" below is an experimental **app** feature, not a
published artifact.

## Published library (embeddings / reranking) — unchanged scope

- MiniLM + E5 (WordPiece `small-v2`/`base-v2`/experimental `large-v2`) embeddings; cross-encoder reranker. CPU +
  DirectML parity. Direct Java 21 API via `directml-runtime`.

## Workbench text generation (experimental, native Java/DirectML — no Python, no ONNX Runtime)

Models load from an internal `.wdmlpack` compiled on **Download → Convert**; inference never compiles. Backends:
**WARP** (D3D12 software rasterizer; dense matmuls on DirectML, rest on CPU), **AUTO** (first hardware adapter else
CPU reference), **CPU** (validated reference; Gemma `Backend=CPU` is the one legacy external Python path).

New / updated runnable models:

- **Gemma 3 270M-it** — native Java/DirectML (WARP/AUTO); resident/batched decode default.
- **Qwen2.5-Coder 0.5B-Instruct** — native DirectML INT4 (`model_q4f16.wdmlpack`).
- **SmolLM2 135M / 360M-Instruct** — native DirectML/WARP dense projections + CPU reference fallback.
- **T5 / Flan-T5 / CodeT5** — the four curated models (`google-t5/t5-small`, `google/flan-t5-small`,
  `Salesforce/codet5-small`, `Salesforce/codet5-base-multi-sum`) are **real-certified** CPU == WARP (greedy; token
  ids + text identical), via a mixed DirectML/WARP path.
- **Phi-3-mini (`microsoft/Phi-3-mini-4k-instruct-onnx`)** — now runnable from `model_phi3.wdmlpack` via a
  **heap-light** package load (qweight/zeropoints reference the mmap'd buffers; ~0.8 GB instead of ~3 GB) and the
  native `Phi3Runtime` (CPU path).

Infrastructure:

- **wdmlpack package path** for the generation families (compiler + `RuntimePackage` loader), mirroring Qwen/T5/SmolLM2.
- **Large wdmlpack reader (>2 GB):** the shared reader reads packages larger than `Integer.MAX_VALUE` (positional
  manifest read + per-tensor windowed mapping); the format is unchanged. This is what lets the ~2.39 GB Phi-3-mini
  package reload.
- **Honest Workbench model status:** runnable models are runnable by status; not-executable models stay PLANNED with a
  clear guard message; no silent Python/ONNX-Runtime fallbacks.

## Known limits

- **Phi-3.5 stays PLANNED** (same architecture; would reuse the Phi-3 compiler/runtime once its weights + download
  are wired). Larger Qwen (1.5B / 3B) stay PLANNED.
- **T5 certification scope:** exactly the four curated models above. Other T5/CodeT5 checkpoints are **not**
  blanket-certified and must run their own gated cert.
- **AUTO** uses a hardware adapter when present but is not uniformly faster/stable everywhere; **CPU/WARP** is the
  always-available path.
- **Real-model tests are opt-in** (e.g. `-Dt5.realModel`, `-Dt5.correctness.cert`, `-Dphi3.realModel`,
  `-Dphi3.compile.realModel`); the standard test run stays light.
- **Memory:** large generation models can need several GB of RAM. Downloaded `model/...` artifacts are git-ignored
  and never committed; model weights are not shipped in Maven artifacts.
- Text generation is **not** a published Maven artifact; it is a Workbench app capability.

See [`SUPPORTED_MODELS.md`](../SUPPORTED_MODELS.md) and [`docs/workbench-model-status.md`](workbench-model-status.md)
for the full matrix.
