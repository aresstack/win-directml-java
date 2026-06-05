package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.model.RuntimeTensor;
import com.aresstack.windirectml.inference.model.SourceTensorDataType;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class SmolLM2ReferenceGenerationLoopTest {

    @Test
    void referenceRuntimeGeneratesTokenIdsFromDenseWeights() {
        SmolLM2ReferenceGenerationLoop generationLoop = new SmolLM2ReferenceGenerationLoop(weightsSelectingTokenOne());

        SmolLM2TokenRuntimeResult result = generationLoop.generate(
                new SmolLM2TokenRuntimeRequest(List.of(0), 1, SmolLM2GenerationOptions.greedy()));

        assertEquals(List.of(1), result.generatedTokenIds());
        assertEquals(List.of(0, 1), result.fullTokenIds());
        assertEquals(1, result.tokensGenerated());
        assertEquals("length", result.finishReason());
    }

    @Test
    void referenceRuntimeStopsOnEosToken() {
        SmolLM2ReferenceGenerationLoop generationLoop = new SmolLM2ReferenceGenerationLoop(weightsSelectingEosToken());

        SmolLM2TokenRuntimeResult result = generationLoop.generate(
                new SmolLM2TokenRuntimeRequest(List.of(0), 4, SmolLM2GenerationOptions.greedy()));

        assertEquals(List.of(2), result.generatedTokenIds());
        assertEquals(1, result.tokensGenerated());
        assertEquals("eos_token", result.finishReason());
    }

    @Test
    void tokenRuntimeRequestKeepsTokenizerOutOfTheReferencePath() {
        SmolLM2TokenRuntimeRequest request = new SmolLM2TokenRuntimeRequest(
                List.of(0), 1, SmolLM2GenerationOptions.greedy());

        assertEquals(List.of(0), request.inputTokenIds());
    }

    private static SmolLM2Weights weightsSelectingTokenOne() {
        return weightsWithLmHead(new float[]{
                0.0f, 0.0f,
                1.0f, 0.0f,
                0.0f, 0.0f,
                0.0f, 1.0f
        });
    }

    private static SmolLM2Weights weightsSelectingEosToken() {
        return weightsWithLmHead(new float[]{
                0.0f, 0.0f,
                0.0f, 0.0f,
                1.0f, 0.0f,
                0.0f, 1.0f
        });
    }

    private static SmolLM2Weights weightsWithLmHead(float[] lmHead) {
        SmolLM2Config config = config();
        Map<SmolLM2TensorRole, SmolLM2WeightTensor> layerTensors = new EnumMap<>(SmolLM2TensorRole.class);
        layerTensors.put(SmolLM2TensorRole.LAYER_INPUT_NORM, rootLayerTensor(SmolLM2TensorRole.LAYER_INPUT_NORM, "input_norm", 0, new float[]{1.0f, 1.0f}, 2));
        layerTensors.put(SmolLM2TensorRole.LAYER_SELF_Q, rootLayerTensor(SmolLM2TensorRole.LAYER_SELF_Q, "q", 0, zeros(4), 2, 2));
        layerTensors.put(SmolLM2TensorRole.LAYER_SELF_K, rootLayerTensor(SmolLM2TensorRole.LAYER_SELF_K, "k", 0, zeros(4), 2, 2));
        layerTensors.put(SmolLM2TensorRole.LAYER_SELF_V, rootLayerTensor(SmolLM2TensorRole.LAYER_SELF_V, "v", 0, zeros(4), 2, 2));
        layerTensors.put(SmolLM2TensorRole.LAYER_SELF_O, rootLayerTensor(SmolLM2TensorRole.LAYER_SELF_O, "o", 0, zeros(4), 2, 2));
        layerTensors.put(SmolLM2TensorRole.LAYER_POST_ATTENTION_NORM, rootLayerTensor(SmolLM2TensorRole.LAYER_POST_ATTENTION_NORM, "post_norm", 0, new float[]{1.0f, 1.0f}, 2));
        layerTensors.put(SmolLM2TensorRole.LAYER_MLP_GATE, rootLayerTensor(SmolLM2TensorRole.LAYER_MLP_GATE, "gate", 0, zeros(4), 2, 2));
        layerTensors.put(SmolLM2TensorRole.LAYER_MLP_UP, rootLayerTensor(SmolLM2TensorRole.LAYER_MLP_UP, "up", 0, zeros(4), 2, 2));
        layerTensors.put(SmolLM2TensorRole.LAYER_MLP_DOWN, rootLayerTensor(SmolLM2TensorRole.LAYER_MLP_DOWN, "down", 0, zeros(4), 2, 2));

        return new SmolLM2Weights(
                config,
                rootTensor(SmolLM2TensorRole.TOKEN_EMBEDDING, "embed", new float[]{
                        1.0f, 0.0f,
                        0.0f, 1.0f,
                        0.0f, 0.0f,
                        1.0f, 1.0f
                }, 4, 2),
                rootTensor(SmolLM2TensorRole.FINAL_NORM, "final_norm", new float[]{1.0f, 1.0f}, 2),
                rootTensor(SmolLM2TensorRole.LM_HEAD, "lm_head", lmHead, 4, 2),
                false,
                List.of(new SmolLM2LayerWeights(0, layerTensors)),
                0L);
    }

    private static SmolLM2Config config() {
        return new SmolLM2Config("llama", List.of("LlamaForCausalLM"),
                2, 2, 1, 1, 1, 2, 4, 8,
                1.0e-6d, 10000.0d, "silu", false, false, 0, 2, null, false);
    }

    private static SmolLM2WeightTensor rootTensor(SmolLM2TensorRole role, String name, float[] values, long... dims) {
        return new SmolLM2WeightTensor(
                new SmolLM2TensorRoleBinding(role, SmolLM2TensorRoleBinding.NO_LAYER, name),
                runtimeTensor(name, values, dims));
    }

    private static SmolLM2WeightTensor rootLayerTensor(SmolLM2TensorRole role, String name, int layerIndex,
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

    private static float[] zeros(int count) {
        return new float[count];
    }
}
