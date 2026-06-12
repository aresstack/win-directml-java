/**
 * Encoder-decoder (seq2seq) model family layer — counterpart to {@code decoderonly} for Qwen/SmolLM2.
 *
 * <p>Home for the shared seq2seq runtime concepts (currently realised by the T5/CodeT5 family in the {@code t5}
 * package). T5 is encoder-decoder: {@code encoder(input) -> hidden states -> decoder step with cross-attention ->
 * lm head}. Its generation loop is therefore structurally different from {@code decoderonly} and is NOT reused;
 * commonality with decoder-only families lives <em>below</em> the loop (WARP dense/fused kernels, runtime package /
 * .wdmlpack lifecycle, prompt pipeline, generation result / streaming, profiling, token-selection / stop handling).
 *
 * <p>Scope note (slice T5-1): this package is the documented boundary for that layer. It deliberately holds no
 * speculative generic interfaces yet — T5 already has working equivalents and is currently the only seq2seq runtime
 * (CodeT5 is a tokenizer variant on the T5 runtime, not a second runtime). Concrete shared seams are introduced only
 * when a real shared building block requires them; see {@code docs/concept-t5-seq2seq-homogenization.md} for the
 * inventory and the planned slice sequence (T5-2: shared WARP dense projection first).
 */
package com.aresstack.windirectml.inference.seq2seq;
