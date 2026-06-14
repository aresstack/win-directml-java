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

## Slice status

| Slice | Status |
|-------|--------|
| GEMMA-WARP-1 family shell + config/inspect | in progress |
| GEMMA-WARP-2 tokenizer + chat template | pending |
| GEMMA-WARP-3 wdmlpack compiler shell | pending |
| GEMMA-WARP-4 CPU reference math | pending |
| GEMMA-WARP-5..8 WARP kernels / single layer | pending (GPU-gated) |
| GEMMA-WARP-9 full prefill forward | pending |
| GEMMA-WARP-10 decode session + KV cache | pending |
| GEMMA-WARP-11 workbench native flag | pending |
| GEMMA-WARP-12 perf/heap comparison | pending |

## Open points / blockers (resolve at the end)

- **GPU/WARP execution not verifiable in this environment:** no DirectML/D3D12 device in the build
  sandbox. WARP-kernel slices (5–10) can be written + given device-free tests (masks, layouts,
  reference parity), but actual on-GPU numerical validation must run on a Windows host with a GPU
  (mirrors `SmolLM2NativeWarpExecutorTest`, which is `@EnabledIf(WindowsBindings.isSupported())`).
  Strategy: build a **CPU reference** that is verifiable against HF/transformers, then mirror it on
  WARP and gate the GPU tests.
- (further blockers appended as encountered)
