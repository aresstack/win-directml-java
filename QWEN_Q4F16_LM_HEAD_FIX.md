# Qwen q4f16 LM head fix

This patch fixes the q4f16 startup failure:

```text
LM head quantized weight not found in ONNX graph
```

`onnx-community/Qwen2.5-Coder-0.5B-Instruct/config.json` declares `tie_word_embeddings: true`.
For that layout the output projection is tied to `model.embed_tokens.weight`, so the runtime must not require a separate
quantized `lm_head` MatMulNBits node just because the decoder layers are quantized.

## Changed

- `Qwen2Weights` now honors `tie_word_embeddings=true` for quantized models.
- For quantized/tied models, `lm_head` is created from `embed_tokens` instead of failing.
- By default the tied `lm_head` now uses the exact dense embedding projection. Use `-Dqwen.tiedLmHead.quantize=true`
  only for an explicit experimental bandwidth benchmark.
- Explicit `lm_head` loading remains available for untied models.
- The Workbench model selector and downloader behavior from the previous patch is kept.

## Runtime switch

Enable experimental runtime quantization of the tied `lm_head` with:

```text
-Dqwen.tiedLmHead.quantize=true
```

## Expected log

For `model_q4f16.onnx` you should see the model format marker and then the tied head handling:

```text
Model format: INT4 quantized (MatMulNBits)
Using tied lm_head (tie_word_embeddings=true): reusing embed_tokens [151936, 896]
Loaded lm_head: [151936, 896]
```

The decoder layers still use the quantized MatMulNBits path. The tied `lm_head` is no longer expected as a separate ONNX
graph weight; runtime quantization of the tied head is now opt-in.
