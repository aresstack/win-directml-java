package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.windows.OnnxModelReader;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class T5EncoderPipelineTest {

    @TempDir
    Path tempDir;

    @Test
    void encodesInputTokensWithReferencePipeline() throws Exception {
        T5Config config = T5TestFixtures.tinyConfig(false);
        Path pack = compilePack(config, referenceTensors(config));
        T5Runtime runtime = T5Runtime.load(T5RuntimePackage.open(pack));

        T5EncoderOutput output = runtime.encode(new int[]{1, 2, 3});

        assertEquals(3, output.inputTokens());
        assertEquals(config.modelSize(), output.hiddenSize());
        assertEquals(3 * config.modelSize(), output.hiddenStates().length);
        assertArrayEquals(new boolean[]{true, true, true}, output.attentionMask());
        assertAllFinite(output.hiddenStates());
    }

    @Test
    void defaultMaskTreatsPadTokensAsInactive() throws Exception {
        T5Config config = T5TestFixtures.tinyConfig(false);
        Path pack = compilePack(config, referenceTensors(config));
        T5Runtime runtime = T5Runtime.load(T5RuntimePackage.open(pack));

        T5EncoderOutput output = runtime.encode(new int[]{1, config.padTokenId(), 2});

        assertArrayEquals(new boolean[]{true, false, true}, output.attentionMask());
        assertArrayEquals(new float[]{0.0f, 0.0f, 0.0f, 0.0f}, output.tokenHiddenState(1), 0.0001f);
    }

    @Test
    void supportsExplicitAttentionMask() throws Exception {
        T5Config config = T5TestFixtures.tinyConfig(false);
        Path pack = compilePack(config, referenceTensors(config));
        T5Runtime runtime = T5Runtime.load(T5RuntimePackage.open(pack));

        T5EncoderOutput output = runtime.encode(new int[]{1, config.padTokenId(), 2}, new boolean[]{true, true, false});

        assertArrayEquals(new boolean[]{true, true, false}, output.attentionMask());
        assertAllFinite(output.hiddenStates());
    }

    @Test
    void rejectsInputTokenOutsideVocabulary() throws Exception {
        T5Config config = T5TestFixtures.tinyConfig(false);
        Path pack = compilePack(config, referenceTensors(config));
        T5Runtime runtime = T5Runtime.load(T5RuntimePackage.open(pack));

        Exception error = assertThrows(IllegalArgumentException.class, () -> runtime.encode(new int[]{config.vocabSize()}));

        assertTrue(error.getMessage().contains("outside vocabulary"));
    }

    private Path compilePack(T5Config config, Map<String, OnnxTensor> tensors) throws Exception {
        Path modelDir = tempDir.resolve("model-" + System.nanoTime());
        T5TestFixtures.writeConfig(modelDir, config);
        T5TestFixtures.writeSafeTensors(modelDir, tensors);
        Path output = tempDir.resolve("t5-encoder-" + System.nanoTime() + ".wdmlpack");
        T5WdmlPackCompiler.compile(new T5CompileOptions(modelDir, output, false, false));
        return output;
    }

    private static Map<String, OnnxTensor> referenceTensors(T5Config config) {
        Map<String, OnnxTensor> tensors = new LinkedHashMap<>();
        addEmbedding(tensors, "shared.weight", config.vocabSize(), config.modelSize());
        addVector(tensors, "encoder.final_layer_norm.weight", config.modelSize(), 1.0f);
        addVector(tensors, "decoder.final_layer_norm.weight", config.modelSize(), 1.0f);
        addMatrix(tensors, "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                config.relativeAttentionBuckets(), config.attentionHeads(), 0.0f);
        addMatrix(tensors, "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                config.relativeAttentionBuckets(), config.attentionHeads(), 0.0f);
        addEncoderBlock(tensors, 0, config);
        addDecoderBlock(tensors, 0, config);
        return tensors;
    }

    private static void addEncoderBlock(Map<String, OnnxTensor> tensors, int layer, T5Config config) {
        addVector(tensors, "encoder.block." + layer + ".layer.0.layer_norm.weight", config.modelSize(), 1.0f);
        addVector(tensors, "encoder.block." + layer + ".layer.1.layer_norm.weight", config.modelSize(), 1.0f);
        addAttention(tensors, "encoder.block." + layer + ".layer.0.SelfAttention", config, false);
        addMatrix(tensors, "encoder.block." + layer + ".layer.1.DenseReluDense.wi.weight",
                config.feedForwardSize(), config.modelSize(), 0.0f);
        addMatrix(tensors, "encoder.block." + layer + ".layer.1.DenseReluDense.wo.weight",
                config.modelSize(), config.feedForwardSize(), 0.0f);
    }

    private static void addDecoderBlock(Map<String, OnnxTensor> tensors, int layer, T5Config config) {
        addVector(tensors, "decoder.block." + layer + ".layer.0.layer_norm.weight", config.modelSize(), 1.0f);
        addVector(tensors, "decoder.block." + layer + ".layer.1.layer_norm.weight", config.modelSize(), 1.0f);
        addVector(tensors, "decoder.block." + layer + ".layer.2.layer_norm.weight", config.modelSize(), 1.0f);
        addAttention(tensors, "decoder.block." + layer + ".layer.0.SelfAttention", config, false);
        addAttention(tensors, "decoder.block." + layer + ".layer.1.EncDecAttention", config, false);
        addMatrix(tensors, "decoder.block." + layer + ".layer.2.DenseReluDense.wi.weight",
                config.feedForwardSize(), config.modelSize(), 0.0f);
        addMatrix(tensors, "decoder.block." + layer + ".layer.2.DenseReluDense.wo.weight",
                config.modelSize(), config.feedForwardSize(), 0.0f);
    }

    private static void addAttention(Map<String, OnnxTensor> tensors, String prefix, T5Config config, boolean identity) {
        float value = identity ? 1.0f : 0.0f;
        addMatrix(tensors, prefix + ".q.weight", config.attentionInnerSize(), config.modelSize(), value);
        addMatrix(tensors, prefix + ".k.weight", config.attentionInnerSize(), config.modelSize(), value);
        addMatrix(tensors, prefix + ".v.weight", config.attentionInnerSize(), config.modelSize(), value);
        addMatrix(tensors, prefix + ".o.weight", config.modelSize(), config.attentionInnerSize(), value);
    }

    private static void addEmbedding(Map<String, OnnxTensor> tensors, String name, int rows, int columns) {
        float[] values = new float[rows * columns];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                values[row * columns + column] = row == 0 ? 0.0f : row + column + 1.0f;
            }
        }
        tensors.put(name, tensor(name, new long[]{rows, columns}, values));
    }

    private static void addMatrix(Map<String, OnnxTensor> tensors, String name, int rows, int columns, float value) {
        float[] values = new float[rows * columns];
        for (int i = 0; i < values.length; i++) {
            values[i] = value;
        }
        tensors.put(name, tensor(name, new long[]{rows, columns}, values));
    }

    private static void addVector(Map<String, OnnxTensor> tensors, String name, int length, float value) {
        float[] values = new float[length];
        for (int i = 0; i < values.length; i++) {
            values[i] = value;
        }
        tensors.put(name, tensor(name, new long[]{length}, values));
    }

    private static OnnxTensor tensor(String name, long[] dims, float[] values) {
        ByteBuffer raw = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float value : values) {
            raw.putFloat(value);
        }
        raw.flip();
        return new OnnxTensor(name, dims, OnnxModelReader.ONNX_FLOAT,
                new float[0], new byte[0], raw.asReadOnlyBuffer(), raw.remaining());
    }

    private static void assertAllFinite(float[] values) {
        for (float value : values) {
            assertTrue(Float.isFinite(value), "Expected finite value but got " + value);
        }
    }
}
