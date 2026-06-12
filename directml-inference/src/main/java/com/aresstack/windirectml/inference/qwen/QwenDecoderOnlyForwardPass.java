package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyConfig;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyDecodeSession;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyForwardPass;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyWarpDecodeProfile;

import java.util.Objects;

/**
 * Experimental Qwen forward pass on the shared {@link DecoderOnlyForwardPass} seam.
 *
 * <p>Vends {@link QwenDecoderOnlyDecodeSession}s that drive Qwen's real decode pipeline (via {@link QwenDecodeSteps}),
 * so a future shared generation loop could run Qwen the same way it runs SmolLM2 — while Qwen keeps its own
 * GPU-resident KV cache. This is <b>not</b> wired into the workbench or any default path: it is gated behind
 * {@code -D{@value #EXPERIMENTAL_FLAG}=true} and must be constructed explicitly. The production
 * {@link Qwen2Runtime} remains the default and is unchanged.</p>
 *
 * <p>The {@code maxTokens} hint of {@link #newDecodeSession(int)} is intentionally ignored: Qwen owns and sizes its
 * KV cache internally, which is the whole point of the session seam.</p>
 */
public final class QwenDecoderOnlyForwardPass implements DecoderOnlyForwardPass {

    /** System property that must be {@code true} to construct this experimental path. */
    public static final String EXPERIMENTAL_FLAG = "qwen.decoderonly.experimental";

    private final DecoderOnlyConfig config;
    private final QwenDecodeSteps decodeSteps;
    // Qwen has no decode micro-profile; a disabled profile keeps the shared loop overhead-free.
    private final DecoderOnlyWarpDecodeProfile decodeProfile = new DecoderOnlyWarpDecodeProfile(false, "Qwen");

    /**
     * @param config      Qwen's config (already a {@link DecoderOnlyConfig})
     * @param decodeSteps Qwen's step-wise decode seam (typically a {@link Qwen2Runtime})
     * @throws IllegalStateException if the experimental flag is not enabled
     */
    public QwenDecoderOnlyForwardPass(Qwen2Config config, QwenDecodeSteps decodeSteps) {
        if (!experimentalEnabled()) {
            throw new IllegalStateException(
                    "Qwen decoder-only session path is experimental; enable -D" + EXPERIMENTAL_FLAG + "=true");
        }
        this.config = Objects.requireNonNull(config, "config");
        this.decodeSteps = Objects.requireNonNull(decodeSteps, "decodeSteps");
    }

    /** Whether the experimental Qwen decoder-only session path is enabled via the system property. */
    public static boolean experimentalEnabled() {
        return Boolean.getBoolean(EXPERIMENTAL_FLAG);
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
        // maxTokens ignored on purpose: Qwen owns/sizes its (GPU-resident) KV cache internally.
        return new QwenDecoderOnlyDecodeSession(decodeSteps);
    }
}
