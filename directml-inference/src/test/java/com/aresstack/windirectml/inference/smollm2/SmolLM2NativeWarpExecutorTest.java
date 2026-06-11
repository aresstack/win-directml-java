package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyMath;
import com.aresstack.windirectml.inference.model.RuntimeTensor;
import com.aresstack.windirectml.inference.model.SourceTensorDataType;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Numerical WARP-vs-reference verification for the SmolLM2 native executor.
 *
 * <p>The WARP forward pass must produce the same logits as the CPU reference forward pass (apart from GPU
 * floating-point rounding) for identical input token IDs. This is the bring-up gate before enabling WARP text
 * generation: top-1 token must match and the full logit vector must agree within a small tolerance.</p>
 */
class SmolLM2NativeWarpExecutorTest {

    private static final float TOLERANCE = 2.0e-2f;

    @Test
    void warpForwardPassMatchesReferenceLogits() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML");
        SmolLM2Weights weights = syntheticWeights();
        List<Integer> tokenIds = List.of(3, 7, 1, 5);

        float[] referenceLogits = new SmolLM2ReferenceForwardPass(weights).logitsForLastToken(tokenIds);

        try (WindowsBindings bindings = new WindowsBindings()) {
            bindings.init("warp");
            try (SmolLM2WarpForwardPass warpForwardPass = new SmolLM2WarpForwardPass(bindings, weights)) {
                SmolLM2ReferenceKvCache kvCache = SmolLM2ReferenceKvCache.create(weights.config());
                float[] warpLogits = warpForwardPass.logitsForLastToken(tokenIds, kvCache);

                assertEquals(referenceLogits.length, warpLogits.length);
                assertEquals(DecoderOnlyMath.argmax(referenceLogits), DecoderOnlyMath.argmax(warpLogits),
                        "WARP top-1 token must match the reference forward pass");
                assertArrayEquals(referenceLogits, warpLogits, TOLERANCE);
            }
        }
    }

    @Test
    void nativeExecutorGeneratesSameTokensAsReferenceLoop() {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML");
        SmolLM2Weights weights = syntheticWeights();
        SmolLM2TokenRuntimeRequest request = new SmolLM2TokenRuntimeRequest(
                List.of(3, 7), 6, SmolLM2GenerationOptions.greedy());

        SmolLM2TokenRuntimeResult referenceResult =
                new SmolLM2ReferenceGenerationLoop(weights).generate(request);

        SmolLM2WarpRuntimePlan plan = new SmolLM2WarpRuntimePlanner()
                .plan(weights, weights.config().maxPositionEmbeddings());
        SmolLM2TokenRuntimeResult warpResult =
                new SmolLM2NativeWarpExecutor().generate(weights, plan, request);

        assertEquals(referenceResult.generatedTokenIds(), warpResult.generatedTokenIds(),
                "WARP greedy decoding must match the reference loop token-for-token");
    }

    @Test
    void warpSessionStreamsEachGeneratedTokenToTheConsumer() {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML");
        SmolLM2Weights weights = syntheticWeights();
        SmolLM2TokenRuntimeRequest request = new SmolLM2TokenRuntimeRequest(
                List.of(3, 7), 6, SmolLM2GenerationOptions.greedy());

        SmolLM2WarpRuntimePlan plan = new SmolLM2WarpRuntimePlanner()
                .plan(weights, weights.config().maxPositionEmbeddings());

        List<Integer> streamed = new ArrayList<>();
        SmolLM2TokenRuntimeResult result;
        try (SmolLM2NativeWarpSession session = new SmolLM2NativeWarpSession(weights, plan, "warp")) {
            result = session.generateTokenIds(request, streamed::add);
        }

        // The non-stop tokens (everything except a trailing stop token) must have been streamed in order as they
        // were produced — the WARP path must not buffer the whole batch and skip the per-token callback.
        List<Integer> expectedStreamed = result.generatedTokenIds();
        if (!expectedStreamed.isEmpty() && "eos_token".equals(result.finishReason())) {
            expectedStreamed = expectedStreamed.subList(0, expectedStreamed.size() - 1);
        }
        assertEquals(expectedStreamed, streamed,
                "WARP generate(request, consumer) must invoke onToken for every accepted token");
    }

    // ── Synthetic weights ─────────────────────────────────────────────────

    private static SmolLM2Config config() {
        // hidden=8, heads=2, kvHeads=1, headDim=4 (qSize=8, kvSize=4), intermediate=16, layers=2, vocab=16.
        return new SmolLM2Config("llama", List.of("LlamaForCausalLM"),
                8, 16, 2, 2, 1, 4, 16, 32,
                1.0e-5d, 10000.0d, "silu", false, false, 0, 2, null, false);
    }

    private static SmolLM2Weights syntheticWeights() {
        SmolLM2Config config = config();
        int hidden = config.hiddenSize();
        int qSize = config.numAttentionHeads() * config.effectiveHeadDim();
        int kvSize = config.effectiveKeyValueHeads() * config.effectiveHeadDim();
        int inter = config.intermediateSize();
        int vocab = config.vocabSize();

        List<SmolLM2LayerWeights> layers = new ArrayList<>();
        int seed = 1;
        for (int layer = 0; layer < config.numHiddenLayers(); layer++) {
            Map<SmolLM2TensorRole, SmolLM2WeightTensor> tensors = new EnumMap<>(SmolLM2TensorRole.class);
            tensors.put(SmolLM2TensorRole.LAYER_INPUT_NORM,
                    layerTensor(SmolLM2TensorRole.LAYER_INPUT_NORM, "input_norm", layer, ones(hidden), hidden));
            tensors.put(SmolLM2TensorRole.LAYER_SELF_Q,
                    layerTensor(SmolLM2TensorRole.LAYER_SELF_Q, "q", layer, weights(seed++, qSize * hidden), qSize, hidden));
            tensors.put(SmolLM2TensorRole.LAYER_SELF_K,
                    layerTensor(SmolLM2TensorRole.LAYER_SELF_K, "k", layer, weights(seed++, kvSize * hidden), kvSize, hidden));
            tensors.put(SmolLM2TensorRole.LAYER_SELF_V,
                    layerTensor(SmolLM2TensorRole.LAYER_SELF_V, "v", layer, weights(seed++, kvSize * hidden), kvSize, hidden));
            tensors.put(SmolLM2TensorRole.LAYER_SELF_O,
                    layerTensor(SmolLM2TensorRole.LAYER_SELF_O, "o", layer, weights(seed++, hidden * qSize), hidden, qSize));
            tensors.put(SmolLM2TensorRole.LAYER_POST_ATTENTION_NORM,
                    layerTensor(SmolLM2TensorRole.LAYER_POST_ATTENTION_NORM, "post_norm", layer, ones(hidden), hidden));
            tensors.put(SmolLM2TensorRole.LAYER_MLP_GATE,
                    layerTensor(SmolLM2TensorRole.LAYER_MLP_GATE, "gate", layer, weights(seed++, inter * hidden), inter, hidden));
            tensors.put(SmolLM2TensorRole.LAYER_MLP_UP,
                    layerTensor(SmolLM2TensorRole.LAYER_MLP_UP, "up", layer, weights(seed++, inter * hidden), inter, hidden));
            tensors.put(SmolLM2TensorRole.LAYER_MLP_DOWN,
                    layerTensor(SmolLM2TensorRole.LAYER_MLP_DOWN, "down", layer, weights(seed++, hidden * inter), hidden, inter));
            layers.add(new SmolLM2LayerWeights(layer, tensors));
        }

        return new SmolLM2Weights(
                config,
                rootTensor(SmolLM2TensorRole.TOKEN_EMBEDDING, "embed", weights(101, vocab * hidden), vocab, hidden),
                rootTensor(SmolLM2TensorRole.FINAL_NORM, "final_norm", ones(hidden), hidden),
                rootTensor(SmolLM2TensorRole.LM_HEAD, "lm_head", weights(202, vocab * hidden), vocab, hidden),
                false,
                layers,
                0L);
    }

    private static float[] ones(int count) {
        float[] values = new float[count];
        java.util.Arrays.fill(values, 1.0f);
        return values;
    }

    /** Deterministic small pseudo-random weights in roughly [-0.5, 0.5]. */
    private static float[] weights(int seed, int count) {
        float[] values = new float[count];
        long state = seed * 2654435761L + 1442695040888963407L;
        for (int i = 0; i < count; i++) {
            state = state * 6364136223846793005L + 1442695040888963407L;
            int bits = (int) (state >>> 40);
            values[i] = ((bits & 0xFFFF) / 65535.0f) - 0.5f;
        }
        return values;
    }

    private static SmolLM2WeightTensor rootTensor(SmolLM2TensorRole role, String name, float[] values, long... dims) {
        return new SmolLM2WeightTensor(
                new SmolLM2TensorRoleBinding(role, SmolLM2TensorRoleBinding.NO_LAYER, name),
                runtimeTensor(name, values, dims));
    }

    private static SmolLM2WeightTensor layerTensor(SmolLM2TensorRole role, String name, int layerIndex,
                                                   float[] values, long... dims) {
        return new SmolLM2WeightTensor(
                new SmolLM2TensorRoleBinding(role, layerIndex, name),
                runtimeTensor(name, values, dims));
    }

    private static RuntimeTensor runtimeTensor(String name, float[] values, long... dims) {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float value : values) {
            buffer.putFloat(value);
        }
        buffer.flip();
        return new RuntimeTensor(name, dims, SourceTensorDataType.FLOAT.onnxCode(), buffer, buffer.remaining());
    }
}
