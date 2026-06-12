package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyTokenSelector;

/**
 * Selects the next SmolLM2 token from logits.
 *
 * <p>The SmolLM2 flavour of the shared {@link DecoderOnlyTokenSelector} seam; it adds no methods, it just lets the
 * SmolLM2 sampler factory keep a family-specific type while plugging straight into the shared generation loop.</p>
 */
interface SmolLM2TokenSampler extends DecoderOnlyTokenSelector {
}
