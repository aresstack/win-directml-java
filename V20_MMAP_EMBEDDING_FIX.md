# v20 mmap embedding fix

This patch fixes a v20 regression where the heap-light ONNX parser could retain the
large inline FP16 embedding payload as an mmap slice but fail the exact
`model.embed_tokens.weight` name lookup during Qwen startup.

If the explicit embedding tensor name is not found, Qwen now recovers the single
inline FP16 tensor with the exact `[vocabSize, hiddenSize]` embedding shape and
uses it as the embedding table. The preferred path is still the normal name lookup;
this is only a safe fallback for the mmap-backed inline-tensor parser path.

Expected successful startup logs include either:

```text
Using mmap-backed inline FP16 embedding table for model.embed_tokens.weight
```

or, if the parser lost the initializer key:

```text
Embedding tensor name lookup failed; recovered mmap-backed inline FP16 embedding by shape [151936, 896] from '...'
```
