# Decision: Qwen2.5-Coder Artifact Format

Issue: #96 (parallel research task for Qwen CausalLM epic #94)

## Decision

**Selected artifact format: ONNX export layout (INT4 AWQ block-128)**

Target model: `Qwen/Qwen2.5-Coder-0.5B-Instruct`

Source status: **TBD / research**.

The previously referenced Microsoft repo
`microsoft/Qwen2.5-Coder-0.5B-Instruct-ONNX` is not used as a selected source in
this decision. The currently known public ONNX candidate is
[`onnx-community/Qwen2.5-Coder-0.5B-Instruct`](https://huggingface.co/onnx-community/Qwen2.5-Coder-0.5B-Instruct),
but its Transformers.js-style layout must be evaluated before claiming
DirectML INT4 AWQ block-128 compatibility.

## Rationale

The ONNX INT4 AWQ block-128 export is chosen because:

1. **Existing infrastructure** – the Phi-3 path already loads ONNX graphs via
   `OnnxModelReader`, memory-maps `model.onnx.data`, and reads INT4 quantized
   weights through MatMulNBits nodes. Qwen2.5 candidates use the same operator
   pattern in DirectML-compatible INT4 exports.
2. **Pure Java/DirectML** – no external runtime (llama.cpp, ONNX Runtime,
   Python) is required. Final module assignment is still TBD for the Qwen path.
3. **DirectML compatibility** – the INT4 AWQ layout is the same layout already
   dispatched to DirectML kernels in the Phi-3 GPU pipeline; future GPU work
   requires zero format conversion.
4. **Tokenizer availability** – the ONNX export ships `tokenizer.json`,
   `tokenizer_config.json`, `special_tokens_map.json`, and `config.json`.

## Evaluation of candidate paths

### 1. ONNX export layout (INT4 AWQ block-128) ✓ SELECTED

| Criterion                | Assessment                                                                                                                                          |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| Required files           | `model.onnx`, `model.onnx.data`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` (`added_tokens.json` optional) |
| Tokenizer/template       | `tokenizer.json` (BPE, HF fast-tokenizer); ChatML template in `tokenizer_config.json`                                                               |
| Loading complexity       | Low – reuse `OnnxModelReader`; new `Qwen2Config` record + `Qwen2Weights` loader mirroring `Phi3Weights`                                             |
| CPU memory footprint     | ~0.35 GB for 0.5B INT4 (block-128) external data                                                                                                    |
| Expected CPU latency     | ~30–80 ms/token (single-thread dequant + matmul, similar to Phi-3 CPU path)                                                                         |
| DirectML compatibility   | Direct – same MatMulNBits dispatch as Phi-3 GPU pipeline                                                                                            |
| Phi-3 code compatibility | High – same graph-parse → weight-extract → runtime pattern                                                                                          |
| Pure Java/DirectML       | Yes                                                                                                                                                 |

### 2. Safetensors / HuggingFace checkpoint layout

| Criterion                | Assessment                                                                                                 |
|--------------------------|------------------------------------------------------------------------------------------------------------|
| Required files           | `model.safetensors` (or sharded), `config.json`, `tokenizer.json`                                          |
| Tokenizer/template       | Same as ONNX path                                                                                          |
| Loading complexity       | Medium – requires new Safetensors reader (header-parse + mmap); simpler than ONNX graph parse but new code |
| CPU memory footprint     | ~1 GB FP16 or ~0.5 GB BF16 for 0.5B; no built-in quantization                                              |
| Expected CPU latency     | Slower (FP16/BF16 matmuls without quantization; no INT4 path)                                              |
| DirectML compatibility   | Requires either FP16 dispatch or separate quantization step before GPU upload                              |
| Phi-3 code compatibility | Low – Phi-3 does not use Safetensors; new reader from scratch                                              |
| Pure Java/DirectML       | Yes, but new reader + no quantization story                                                                |

### 3. GGUF layout

| Criterion                | Assessment                                                                             |
|--------------------------|----------------------------------------------------------------------------------------|
| Required files           | Single `.gguf` file (e.g. `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf`)                   |
| Tokenizer/template       | Embedded in GGUF metadata (requires GGUF metadata parser)                              |
| Loading complexity       | High – new GGUF container reader + Q4_K_M / Q5_K_M dequantization kernels              |
| CPU memory footprint     | ~0.35 GB Q4_K_M                                                                        |
| Expected CPU latency     | Comparable if dequant kernels are optimized                                            |
| DirectML compatibility   | None – GGUF quantization formats differ from DirectML MatMulNBits; requires conversion |
| Phi-3 code compatibility | None – entirely new code path                                                          |
| Pure Java/DirectML       | Technically yes, but large new subsystem (GGUF reader + dequant)                       |

### 4. External local runner adapter

| Criterion                | Assessment                                                           |
|--------------------------|----------------------------------------------------------------------|
| Required files           | Depends on runner (e.g. llama.cpp binary + GGUF file)                |
| Tokenizer/template       | Handled by external runner                                           |
| Loading complexity       | Low (delegate to subprocess), but adds process management            |
| CPU memory footprint     | External process memory; not controllable from Java                  |
| Expected CPU latency     | Depends on runner; typically faster (optimized C/C++)                |
| DirectML compatibility   | None from Java side; runner may have its own DirectML/Vulkan support |
| Phi-3 code compatibility | None – different paradigm                                            |
| Pure Java/DirectML       | **No** – introduces external native runtime dependency               |

## Local model directory layout

The local Workbench/runtime directory is flat. The remote Hugging Face candidate
may still store these files under a subdirectory such as
`directml/directml-int4-awq-block-128/`, but the downloader copies the files into
the top-level local model directory:

```
model/qwen2.5-coder-0.5b-directml-int4/
├── config.json
├── model.onnx
├── model.onnx.data          (downloaded separately, ~350 MB)
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── added_tokens.json        (optional)
```

This is the layout expected by `QwenModelDirValidator`, the Workbench download
button, `scripts/download-qwen.ps1`, and the manual smoke-test docs.

## Scale-up candidates

| Model                       | Status             | Notes                                    |
|-----------------------------|--------------------|------------------------------------------|
| Qwen2.5-Coder-0.5B-Instruct | **Initial target** | ~350 MB INT4; fits comfortably in memory |
| Qwen2.5-Coder-1.5B-Instruct | Scale-up candidate | ~1 GB INT4; same format, larger weights  |
| Qwen2.5-Coder-3B-Instruct   | Scale-up candidate | ~2 GB INT4; may require chunked mmap     |

1.5B and 3B are not initial blockers. They use the same artifact format and
can be added once the 0.5B loading/generation smoke test passes.

## Follow-up tasks

1. **Source verification** – confirm a resolvable Hugging Face source with a
   layout compatible with this decision (candidate to evaluate:
   `onnx-community/Qwen2.5-Coder-0.5B-Instruct`).
2. **Download script / Workbench download** – keep the local flat layout above
   while evaluating the remote source candidate layout.
3. **Smoke test** – End-to-end generation test with the real 0.5B checkpoint.
4. **Promotion decision** – only move Qwen from planned to experimental after
   a real loading/generation smoke test passes.

## Constraints

- No model is marked shipped before a real loading/generation smoke test exists.
- The ONNX graph is used only as a **weight container** (same as Phi-3); the
  runtime does not execute the ONNX graph.
- The selected path keeps the project pure Java/DirectML with no external
  native dependencies.
