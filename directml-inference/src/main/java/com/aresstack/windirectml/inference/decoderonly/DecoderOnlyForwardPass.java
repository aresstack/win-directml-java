package com.aresstack.windirectml.inference.decoderonly;

import java.util.List;

/**
 * Logits source consumed by the shared {@link DecoderOnlyGenerationLoop}.
 *
 * <p>Abstracts the decoder-only forward pass so the generation loop depends on behaviour, not on the concrete
 * GPU-backed {@link DecoderOnlyWarpForwardPass}. This keeps the loop unit-testable without a WARP device and is the
 * seam a second decoder-only family (e.g. Qwen) plugs into to reuse the same loop.</p>
 */
public interface DecoderOnlyForwardPass {

    /** Family-neutral shape view, used by the loop to size the KV cache. */
    DecoderOnlyConfig config();

    /** Decode timing accumulator (enabled flag + label decided by the family). */
    DecoderOnlyWarpDecodeProfile decodeProfile();

    /** LM-head projection time (ns) measured during the most recent {@link #logitsForLastToken} call. */
    long lastCallLmHeadNanos();

    /** Decode the supplied token sequence with the incremental KV cache and return the last token's logits. */
    float[] logitsForLastToken(List<Integer> tokenIds, DecoderOnlyWarpKvCache kvCache);
}
