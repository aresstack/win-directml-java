# Qwen q4f16 output fix

This patch addresses the q4f16 case where the model loads successfully but decodes random-looking text with frequent FIM
tokens.

## Changed

- Preserve Qwen2.5 Q/K/V attention projection biases in the quantized `MatMulNBits` path.
- Resolve quantized Q/K/V biases from direct tensor names and from the `Add` node connected to each `MatMulNBits`
  output.
- Normalize `MatMulNBits` zero-points before handing them to the runtime kernels.
    - Accept packed UINT8/INT8 zero-points.
    - Accept unpacked FLOAT16/FLOAT zero-points.
    - Repack ONNX row-padded zero-points from `[N, ceil(k_blocks * 4 / 8)]` into the runtime's continuous nibble stream.
- Change the tied `lm_head` default back to the exact dense embedding projection.
    - Use `-Dqwen.tiedLmHead.quantize=true` only for an explicit experimental bandwidth benchmark.

## Expected q4f16 log

```text
Qwen ONNX file: model_q4f16.onnx
Model format: INT4 quantized (MatMulNBits)
Attention biases found (q=896, k=128, v=128)
Using tied lm_head (tie_word_embeddings=true): reusing embed_tokens [151936, 896]
Loaded lm_head: [151936, 896]
```

## Optional experimental switch

```text
-Dqwen.tiedLmHead.quantize=true
```

Enable this only after the dense tied `lm_head` produces sane output. It reintroduces the runtime-quantized output
projection and is useful for measuring the remaining memory-bandwidth trade-off.
