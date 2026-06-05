package com.aresstack.windirectml.inference.t5;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class T5CrossAttentionMemoryTest {

    @Test
    void precomputesEncoderKeyValueOnlyOnceForRepeatedDecoderApplications() {
        T5PackageMetadata metadata = T5PackageMetadata.from(T5TestFixtures.tinyConfig(false));
        CountingProjection query = new CountingProjection("q", 4, 4);
        CountingProjection key = new CountingProjection("k", 4, 4);
        CountingProjection value = new CountingProjection("v", 4, 4);
        CountingProjection output = new CountingProjection("o", 4, 4);
        T5CrossAttention attention = new T5CrossAttention(metadata, query, key, value, output);
        T5EncoderOutput encoderOutput = new T5EncoderOutput(3, 4, new float[]{
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f
        }, new boolean[]{true, true, true});

        T5CrossAttentionMemory memory = attention.prepareMemory(encoderOutput);
        attention.apply(new float[]{1.0f, 0.0f, 0.0f, 0.0f}, 1, memory);
        attention.apply(new float[]{0.0f, 1.0f, 0.0f, 0.0f}, 1, memory);

        assertEquals(2, query.sequenceCalls);
        assertEquals(1, key.sequenceCalls);
        assertEquals(1, value.sequenceCalls);
        assertEquals(2, output.sequenceCalls);
    }

    @Test
    void cachedMemoryKeepsSameResultAsDirectCrossAttention() {
        T5PackageMetadata metadata = T5PackageMetadata.from(T5TestFixtures.tinyConfig(false));
        CountingProjection query = new CountingProjection("q", 4, 4);
        CountingProjection key = new CountingProjection("k", 4, 4);
        CountingProjection value = new CountingProjection("v", 4, 4);
        CountingProjection output = new CountingProjection("o", 4, 4);
        T5CrossAttention attention = new T5CrossAttention(metadata, query, key, value, output);
        T5EncoderOutput encoderOutput = new T5EncoderOutput(2, 4, new float[]{
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f
        }, new boolean[]{true, true});
        float[] decoderHidden = new float[]{1.0f, 2.0f, 3.0f, 4.0f};

        float[] direct = attention.apply(decoderHidden, 1, encoderOutput);
        float[] cached = attention.apply(decoderHidden, 1, attention.prepareMemory(encoderOutput));

        assertArrayEquals(direct, cached, 0.0001f);
    }

    private static final class CountingProjection implements T5LinearProjection {
        private final String name;
        private final int inputSize;
        private final int outputSize;
        private int sequenceCalls;

        private CountingProjection(String name, int inputSize, int outputSize) {
            this.name = name;
            this.inputSize = inputSize;
            this.outputSize = outputSize;
        }

        @Override
        public String name() {
            return name;
        }

        @Override
        public int inputSize() {
            return inputSize;
        }

        @Override
        public int outputSize() {
            return outputSize;
        }

        @Override
        public float[] apply(float[] input) {
            float[] result = new float[outputSize];
            System.arraycopy(input, 0, result, 0, Math.min(input.length, result.length));
            return result;
        }

        @Override
        public float[] applySequence(float[] input, int sequenceLength, int inputSize) {
            sequenceCalls++;
            return T5LinearProjection.super.applySequence(input, sequenceLength, inputSize);
        }
    }
}
