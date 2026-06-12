package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyConfig;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyDecodeSession;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyForwardPass;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyWarpDecodeProfile;

import java.util.Objects;

/**
 * Adapts the SmolLM2 CPU {@link SmolLM2ReferenceForwardPass} to the shared {@link DecoderOnlyForwardPass} seam so the
 * family-neutral {@link com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGenerationLoop} can drive the
 * reference path exactly like the WARP path — removing SmolLM2's second, family-local generation loop (todo item 4).
 *
 * <p>The decode micro-profile is disabled (the CPU reference uses its own {@link SmolLM2ReferenceForwardProfile}
 * hotspot timings, surfaced separately by the reference loop adapter), so the shared loop skips the WARP decode
 * breakdown for the reference path — matching the previous reference behaviour.</p>
 */
final class SmolLM2ReferenceDecoderOnlyForwardPass implements DecoderOnlyForwardPass {

    private final SmolLM2ReferenceForwardPass forwardPass;
    private final DecoderOnlyConfig config;
    private final DecoderOnlyWarpDecodeProfile decodeProfile;

    SmolLM2ReferenceDecoderOnlyForwardPass(SmolLM2ReferenceForwardPass forwardPass) {
        this.forwardPass = Objects.requireNonNull(forwardPass, "forwardPass");
        this.config = forwardPass.config().toDecoderOnlyConfig();
        this.decodeProfile = new DecoderOnlyWarpDecodeProfile(false, "SmolLM2-reference");
    }

    @Override
    public DecoderOnlyConfig config() {
        return config;
    }

    @Override
    public DecoderOnlyWarpDecodeProfile decodeProfile() {
        return decodeProfile;
    }

    @Override
    public DecoderOnlyDecodeSession newDecodeSession(int maxTokens) {
        return new SmolLM2ReferenceDecodeSession(forwardPass);
    }
}
