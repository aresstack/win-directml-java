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

class T5GenerationLoopTest {

    @TempDir
    Path tempDir;

    @Test
    void generatesStopTokenWithGreedyLmHead() throws Exception {
        T5Config config = T5TestFixtures.untiedConfig();
        T5Runtime runtime = runtime(config, tensorsThatPreferEos(config));
        T5RuntimeRequest request = T5RuntimeRequest.greedy(new int[]{1, 2, 3}, 4, config.specialTokens());

        T5RuntimeResult result = runtime.generate(request);

        assertEquals(T5RuntimeResult.FinishReason.stop_token, result.finishReason());
        assertEquals(1, result.generatedTokens());
        assertArrayEquals(new int[]{config.eosTokenId()}, result.outputTokenIds());
    }

    @Test
    void stopsAtMaxTokensWhenPolicyDoesNotStop() throws Exception {
        T5Config config = T5TestFixtures.untiedConfig();
        T5Runtime runtime = runtime(config, tensorsThatPreferEos(config));
        T5RuntimeRequest request = new T5RuntimeRequest(new int[]{1, 2, 3}, 3,
                T5StopTokenPolicy.neverStop(), config.decoderStartTokenId(), 0.0f, 0);

        T5RuntimeResult result = runtime.generate(request);

        assertEquals(T5RuntimeResult.FinishReason.max_tokens, result.finishReason());
        assertEquals(3, result.generatedTokens());
        assertArrayEquals(new int[]{config.eosTokenId(), config.eosTokenId(), config.eosTokenId()}, result.outputTokenIds());
    }

    @Test
    void acceptsCustomLogitProjector() throws Exception {
        T5Config config = T5TestFixtures.untiedConfig();
        T5Runtime runtime = runtime(config, tensorsThatPreferEos(config));
        T5GenerationLoop loop = T5GenerationLoop.greedy(runtime.encoderPipeline(), runtime.decoderPipeline(),
                new T5LogitProjector() {
                    @Override
                    public float[] logits(float[] decoderHiddenState) {
                        float[] logits = new float[config.vocabSize()];
                        logits[config.eosTokenId()] = 42.0f;
                        return logits;
                    }

                    @Override
                    public int vocabularySize() {
                        return config.vocabSize();
                    }
                });
        T5RuntimeRequest request = T5RuntimeRequest.greedy(new int[]{1, 2}, 4, config.specialTokens());

        T5RuntimeResult result = loop.generate(request);

        assertEquals(T5RuntimeResult.FinishReason.stop_token, result.finishReason());
        assertArrayEquals(new int[]{config.eosTokenId()}, result.outputTokenIds());
    }

    @Test
    void acceptsCustomDecoderRunnerBoundary() {
        T5EncoderRunner encoderRunner = new T5EncoderRunner() {
            @Override
            public T5EncoderOutput encode(int[] inputTokenIds) {
                return encode(inputTokenIds, null);
            }

            @Override
            public T5EncoderOutput encode(int[] inputTokenIds, boolean[] attentionMask) {
                return new T5EncoderOutput(1, 2, new float[]{1.0f, 0.0f}, new boolean[]{true});
            }

            @Override
            public String executionMode() {
                return "test-encoder";
            }
        };
        T5DecoderRunner decoderRunner = new T5DecoderRunner() {
            @Override
            public String executionMode() {
                return "test-decoder";
            }

            @Override
            public T5DecoderState decode(int[] decoderInputIds, T5EncoderOutput encoderOutput) {
                return new T5DecoderState(decoderInputIds.length, 2, new float[]{0.0f, 0.0f});
            }

            @Override
            public T5DecoderState decodeStep(int decoderTokenId, T5EncoderOutput encoderOutput, T5DecoderCache cache) {
                return new T5DecoderState(1, 2, new float[]{0.0f, 1.0f});
            }
        };
        T5LogitProjector projector = new T5LogitProjector() {
            @Override
            public float[] logits(float[] decoderHiddenState) {
                return new float[]{0.0f, 5.0f, 0.0f};
            }

            @Override
            public int vocabularySize() {
                return 3;
            }
        };
        T5GenerationLoop loop = T5GenerationLoop.greedy(encoderRunner, decoderRunner, projector);

        T5RuntimeResult result = loop.generate(new T5RuntimeRequest(new int[]{7}, 1,
                T5StopTokenPolicy.stopAtEos(1), 0, 0.0f, 0));

        assertEquals(T5RuntimeResult.FinishReason.stop_token, result.finishReason());
        assertArrayEquals(new int[]{1}, result.outputTokenIds());
    }


    @Test
    void textGenerationDoesNotStopOnFirstEosOrEmitDecoderStartToken() throws Exception {
        T5Config config = T5TestFixtures.untiedConfig();
        T5Runtime runtime = runtime(config, tensorsThatPreferEos(config));
        T5RuntimeRequest request = T5RuntimeRequest.greedyText(new int[]{1, 2, 3}, 2, config.specialTokens());

        T5RuntimeResult result = runtime.generate(request);

        assertEquals(T5RuntimeResult.FinishReason.stop_token, result.finishReason());
        assertEquals(2, result.generatedTokens());
        assertNotEquals(config.decoderStartTokenId(), result.outputTokenIds()[0]);
        assertEquals(config.eosTokenId(), result.outputTokenIds()[1]);
    }

    @Test
    void rejectsSamplingOptionsInReferenceGeneration() throws Exception {
        T5Config config = T5TestFixtures.untiedConfig();
        T5Runtime runtime = runtime(config, tensorsThatPreferEos(config));
        T5RuntimeRequest request = new T5RuntimeRequest(new int[]{1}, 1,
                T5StopTokenPolicy.neverStop(), config.decoderStartTokenId(), 0.7f, 0);

        Exception error = assertThrows(IllegalArgumentException.class, () -> runtime.generate(request));

        assertTrue(error.getMessage().contains("greedy"));
    }

    private T5Runtime runtime(T5Config config, Map<String, OnnxTensor> tensors) throws Exception {
        Path modelDir = tempDir.resolve("model-" + System.nanoTime());
        T5TestFixtures.writeConfig(modelDir, config);
        T5TestFixtures.writeSafeTensors(modelDir, tensors);
        Path output = tempDir.resolve("t5-generation-" + System.nanoTime() + ".wdmlpack");
        T5WdmlPackCompiler.compile(new T5CompileOptions(modelDir, output, false, false));
        return T5Runtime.load(T5RuntimePackage.open(output));
    }

    private static Map<String, OnnxTensor> tensorsThatPreferEos(T5Config config) {
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
        addLmHeadThatPrefersEos(tensors, config);
        return tensors;
    }

    private static void addEncoderBlock(Map<String, OnnxTensor> tensors, int layer, T5Config config) {
        addVector(tensors, "encoder.block." + layer + ".layer.0.layer_norm.weight", config.modelSize(), 1.0f);
        addVector(tensors, "encoder.block." + layer + ".layer.1.layer_norm.weight", config.modelSize(), 1.0f);
        addAttention(tensors, "encoder.block." + layer + ".layer.0.SelfAttention", config);
        addMatrix(tensors, "encoder.block." + layer + ".layer.1.DenseReluDense.wi.weight",
                config.feedForwardSize(), config.modelSize(), 0.0f);
        addMatrix(tensors, "encoder.block." + layer + ".layer.1.DenseReluDense.wo.weight",
                config.modelSize(), config.feedForwardSize(), 0.0f);
    }

    private static void addDecoderBlock(Map<String, OnnxTensor> tensors, int layer, T5Config config) {
        addVector(tensors, "decoder.block." + layer + ".layer.0.layer_norm.weight", config.modelSize(), 1.0f);
        addVector(tensors, "decoder.block." + layer + ".layer.1.layer_norm.weight", config.modelSize(), 1.0f);
        addVector(tensors, "decoder.block." + layer + ".layer.2.layer_norm.weight", config.modelSize(), 1.0f);
        addAttention(tensors, "decoder.block." + layer + ".layer.0.SelfAttention", config);
        addAttention(tensors, "decoder.block." + layer + ".layer.1.EncDecAttention", config);
        addMatrix(tensors, "decoder.block." + layer + ".layer.2.DenseReluDense.wi.weight",
                config.feedForwardSize(), config.modelSize(), 0.0f);
        addMatrix(tensors, "decoder.block." + layer + ".layer.2.DenseReluDense.wo.weight",
                config.modelSize(), config.feedForwardSize(), 0.0f);
    }

    private static void addAttention(Map<String, OnnxTensor> tensors, String prefix, T5Config config) {
        addMatrix(tensors, prefix + ".q.weight", config.attentionInnerSize(), config.modelSize(), 0.0f);
        addMatrix(tensors, prefix + ".k.weight", config.attentionInnerSize(), config.modelSize(), 0.0f);
        addMatrix(tensors, prefix + ".v.weight", config.attentionInnerSize(), config.modelSize(), 0.0f);
        addMatrix(tensors, prefix + ".o.weight", config.modelSize(), config.attentionInnerSize(), 0.0f);
    }

    private static void addEmbedding(Map<String, OnnxTensor> tensors, String name, int rows, int columns) {
        float[] values = new float[rows * columns];
        values[0] = 1.0f;
        for (int row = 1; row < rows; row++) {
            values[row * columns] = 0.25f;
        }
        tensors.put(name, tensor(name, new long[]{rows, columns}, values));
    }

    private static void addLmHeadThatPrefersEos(Map<String, OnnxTensor> tensors, T5Config config) {
        float[] values = new float[config.vocabSize() * config.modelSize()];
        values[config.eosTokenId() * config.modelSize()] = 10.0f;
        tensors.put("lm_head.weight", tensor("lm_head.weight",
                new long[]{config.vocabSize(), config.modelSize()}, values));
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
}
