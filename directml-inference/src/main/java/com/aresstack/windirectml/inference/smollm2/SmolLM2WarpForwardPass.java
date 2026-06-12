package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyReferenceProjection;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyRotaryEmbedding;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyWarpDecodeProfile;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyWarpDenseProjection;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyWarpForwardPass;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyWarpFusedDenseProjection;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyWarpKvCache;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyWarpLayer;
import com.aresstack.windirectml.windows.WindowsBindings;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * SmolLM2 adapter over the shared {@link DecoderOnlyWarpForwardPass}.
 *
 * <p>This class only resolves SmolLM2's weight tensors into the family-neutral building blocks the generic decoder-only
 * WARP pass consumes (config view, rotary table, token embedding, per-layer fused projections, final norm and LM head)
 * and then delegates all execution. The numerical structure (RMSNorm, RoPE, grouped-query attention, SwiGLU MLP,
 * tied/untied LM head) and the GPU-resident MLP block all live in {@link DecoderOnlyWarpForwardPass}; behaviour is
 * unchanged from the previous SmolLM2-specific forward pass.</p>
 *
 * <p>The projections are uploaded once at construction and reused across every decode step. The LM head runs on WARP
 * by default; the CPU-SIMD reference dense path is only used by the dev/diagnostic runtime, never to optimize the WARP
 * path.</p>
 */
final class SmolLM2WarpForwardPass implements AutoCloseable {

    /**
     * Dev/diagnostic switch: {@code -Dwindirectml.smollm2.lmhead=reference} runs the LM head on the CPU-SIMD reference
     * dense path instead of WARP. This is NOT part of the WARP product path — for {@code backend=WARP} the LM head
     * runs on WARP by default like every other large tensor operation. Any other value keeps the WARP path.
     */
    private static final boolean LM_HEAD_REFERENCE =
            "reference".equalsIgnoreCase(System.getProperty("windirectml.smollm2.lmhead", "warp"));

    private final SmolLM2Config config;
    private final DecoderOnlyWarpForwardPass delegate;

    SmolLM2WarpForwardPass(WindowsBindings windowsBindings, SmolLM2Weights weights) {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        SmolLM2ReferenceWeights referenceWeights = SmolLM2ReferenceWeights.from(weights);
        this.config = referenceWeights.config();

        // Build the rotary table from the SmolLM2 config's full-precision rope_theta (double) before the
        // DecoderOnlyConfig contract narrows it to float, so the precomputed cos/sin tables are unchanged.
        DecoderOnlyRotaryEmbedding rotaryEmbedding = new DecoderOnlyRotaryEmbedding(
                config.effectiveHeadDim(), config.ropeTheta(), config.maxPositionEmbeddings());
        float[] finalNorm = referenceWeights.finalNorm().copyValues();
        SmolLM2DenseTensor tokenEmbedding = referenceWeights.tokenEmbedding();

        List<DecoderOnlyWarpLayer> builtLayers = new ArrayList<>(referenceWeights.layers().size());
        DecoderOnlyWarpDenseProjection builtLmHead = null;
        boolean handedOff = false;
        try {
            for (SmolLM2ReferenceLayerWeights layer : referenceWeights.layers()) {
                builtLayers.add(buildLayer(windowsBindings, layer));
            }
            SmolLM2DenseTensor lmHeadSource = referenceWeights.lmHeadTiedToEmbedding()
                    ? referenceWeights.tokenEmbedding()
                    : referenceWeights.lmHead();
            DecoderOnlyReferenceProjection lmHeadReference = null;
            if (LM_HEAD_REFERENCE) {
                // Dev/diagnostic only: CPU-SIMD reference LM head (explicitly opted in via system property).
                lmHeadReference = lmHeadSource::multiplyVector;
            } else {
                // WARP product path: the LM head runs on WARP like every other large projection.
                builtLmHead = projection(windowsBindings, "lm_head", lmHeadSource);
            }
            DecoderOnlyWarpDecodeProfile profile = new DecoderOnlyWarpDecodeProfile(
                    Boolean.getBoolean("smollm2.profile.decode"), "SmolLM2");
            this.delegate = new DecoderOnlyWarpForwardPass(
                    windowsBindings,
                    config.toDecoderOnlyConfig(),
                    rotaryEmbedding,
                    tokenEmbedding::copyRowInto,
                    builtLayers,
                    finalNorm,
                    builtLmHead,
                    lmHeadReference,
                    profile);
            handedOff = true;
        } finally {
            // Until the generic pass takes ownership, the partially built layers / LM head are ours to release.
            if (!handedOff) {
                for (DecoderOnlyWarpLayer layer : builtLayers) {
                    layer.close();
                }
                if (builtLmHead != null) {
                    builtLmHead.close();
                }
            }
        }
    }

    DecoderOnlyWarpDecodeProfile decodeProfile() {
        return delegate.decodeProfile();
    }

    /** The shared generic forward pass this adapter wraps; used to drive the shared generation loop. */
    DecoderOnlyWarpForwardPass delegate() {
        return delegate;
    }

    SmolLM2Config config() {
        return config;
    }

    /**
     * LM-head projection time (nanoseconds) measured during the most recent {@link #logitsForLastToken} call.
     */
    long lastCallLmHeadNanos() {
        return delegate.lastCallLmHeadNanos();
    }

    /**
     * Decode the supplied token sequence with an incremental KV cache and return the logits for the last token.
     * Delegates to {@link DecoderOnlyWarpForwardPass#logitsForLastToken(List, DecoderOnlyWarpKvCache)}.
     */
    float[] logitsForLastToken(List<Integer> tokenIds, DecoderOnlyWarpKvCache kvCache) {
        return delegate.logitsForLastToken(tokenIds, kvCache);
    }

    @Override
    public void close() {
        delegate.close();
    }

    private static DecoderOnlyWarpLayer buildLayer(WindowsBindings windowsBindings,
                                                   SmolLM2ReferenceLayerWeights layer) {
        int index = layer.layerIndex();
        DecoderOnlyWarpFusedDenseProjection qkv = null;
        DecoderOnlyWarpDenseProjection o = null;
        DecoderOnlyWarpFusedDenseProjection gateUp = null;
        DecoderOnlyWarpDenseProjection down = null;
        try {
            qkv = fusedProjection(windowsBindings, "layer." + index + ".qkv_proj",
                    layer.queryProjection(), layer.keyProjection(), layer.valueProjection());
            o = projection(windowsBindings, "layer." + index + ".o_proj", layer.outputProjection());
            gateUp = fusedProjection(windowsBindings, "layer." + index + ".gate_up_proj",
                    layer.gateProjection(), layer.upProjection());
            down = projection(windowsBindings, "layer." + index + ".down_proj", layer.downProjection());
            return new DecoderOnlyWarpLayer(
                    layer.inputNorm().copyValues(),
                    layer.postAttentionNorm().copyValues(),
                    qkv, o, gateUp, down);
        } catch (RuntimeException e) {
            if (down != null) {
                down.close();
            }
            if (gateUp != null) {
                gateUp.close();
            }
            if (o != null) {
                o.close();
            }
            if (qkv != null) {
                qkv.close();
            }
            throw e;
        }
    }

    private static DecoderOnlyWarpDenseProjection projection(WindowsBindings windowsBindings,
                                                             String name,
                                                             SmolLM2DenseTensor tensor) {
        if (tensor.rank() != 2) {
            throw new IllegalStateException("SmolLM2 WARP projection '" + name + "' must be rank-2 but was rank "
                    + tensor.rank());
        }
        int outputSize = tensor.dim(0);
        int inputSize = tensor.dim(1);
        return DecoderOnlyWarpDenseProjection.fromRowMajorWeights(
                windowsBindings, name, outputSize, inputSize, tensor.copyValues());
    }

    private static DecoderOnlyWarpFusedDenseProjection fusedProjection(WindowsBindings windowsBindings,
                                                                       String name,
                                                                       SmolLM2DenseTensor... tensors) {
        int inputSize = -1;
        List<DecoderOnlyWarpFusedDenseProjection.Part> parts = new ArrayList<>(tensors.length);
        for (SmolLM2DenseTensor tensor : tensors) {
            if (tensor.rank() != 2) {
                throw new IllegalStateException("SmolLM2 WARP fused projection '" + name + "' part '" + tensor.name()
                        + "' must be rank-2 but was rank " + tensor.rank());
            }
            int partOutputSize = tensor.dim(0);
            int partInputSize = tensor.dim(1);
            if (inputSize == -1) {
                inputSize = partInputSize;
            } else if (inputSize != partInputSize) {
                throw new IllegalStateException("SmolLM2 WARP fused projection '" + name
                        + "' parts must share input width: expected " + inputSize + " but got " + partInputSize);
            }
            parts.add(new DecoderOnlyWarpFusedDenseProjection.Part(tensor.name(), partOutputSize, tensor.copyValues()));
        }
        return DecoderOnlyWarpFusedDenseProjection.fromRowMajorParts(windowsBindings, name, inputSize, parts);
    }
}
