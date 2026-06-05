package com.aresstack.windirectml.inference.t5;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

final class T5FeedForwardTest {
    @Test
    void appliesDenseFeedForwardAsSequenceProjection() {
        T5PackageMetadata metadata = metadata("relu");
        RecordingProjection wi = new RecordingProjection("wi", 2, 3,
                new float[][]{
                        {1.0f, 0.0f},
                        {0.0f, 1.0f},
                        {1.0f, 1.0f}
                });
        RecordingProjection wo = new RecordingProjection("wo", 3, 2,
                new float[][]{
                        {1.0f, 0.0f, 0.0f},
                        {0.0f, 1.0f, 1.0f}
                });

        T5FeedForward feedForward = new T5FeedForward(metadata, wi, null, null, wo);
        float[] result = feedForward.apply(new float[]{1.0f, 2.0f, -3.0f, 4.0f}, 2);

        assertArrayEquals(new float[]{1.0f, 5.0f, 0.0f, 5.0f}, result, 0.0001f);
        assertEquals(1, wi.sequenceCalls);
        assertEquals(0, wi.singleCalls);
        assertEquals(1, wo.sequenceCalls);
        assertEquals(0, wo.singleCalls);
    }

    @Test
    void appliesGatedFeedForwardAsSequenceProjection() {
        T5PackageMetadata metadata = metadata("relu");
        RecordingProjection wi0 = new RecordingProjection("wi0", 2, 2,
                new float[][]{
                        {1.0f, 0.0f},
                        {0.0f, 1.0f}
                });
        RecordingProjection wi1 = new RecordingProjection("wi1", 2, 2,
                new float[][]{
                        {2.0f, 0.0f},
                        {0.0f, 3.0f}
                });
        RecordingProjection wo = new RecordingProjection("wo", 2, 2,
                new float[][]{
                        {1.0f, 0.0f},
                        {0.0f, 1.0f}
                });

        T5FeedForward feedForward = new T5FeedForward(metadata, null, wi0, wi1, wo);
        float[] result = feedForward.apply(new float[]{2.0f, 4.0f, -2.0f, 5.0f}, 2);

        assertArrayEquals(new float[]{8.0f, 48.0f, 0.0f, 75.0f}, result, 0.0001f);
        assertEquals(1, wi0.sequenceCalls);
        assertEquals(1, wi1.sequenceCalls);
        assertEquals(1, wo.sequenceCalls);
        assertEquals(0, wi0.singleCalls + wi1.singleCalls + wo.singleCalls);
    }

    private static T5PackageMetadata metadata(String feedForwardProjection) {
        T5Config config = new T5Config(java.util.List.of("T5ForConditionalGeneration"), "t5", true,
                2, 1, 4, 1, 1, 1, 16, 32, 128, 1.0e-6f, 0, 1, 0, true,
                feedForwardProjection);
        return T5PackageMetadata.from(config);
    }

    private static final class RecordingProjection implements T5LinearProjection {
        private final String name;
        private final int inputSize;
        private final int outputSize;
        private final float[][] weight;
        private int singleCalls;
        private int sequenceCalls;

        private RecordingProjection(String name, int inputSize, int outputSize, float[][] weight) {
            this.name = name;
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            this.weight = weight;
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
            singleCalls++;
            return multiply(input);
        }

        @Override
        public float[] applySequence(float[] input, int sequenceLength, int inputSize) {
            sequenceCalls++;
            if (inputSize != this.inputSize) {
                throw new IllegalArgumentException("inputSize mismatch");
            }
            float[] result = new float[sequenceLength * outputSize];
            for (int token = 0; token < sequenceLength; token++) {
                float[] row = new float[inputSize];
                System.arraycopy(input, token * inputSize, row, 0, inputSize);
                float[] projected = multiply(row);
                System.arraycopy(projected, 0, result, token * outputSize, outputSize);
            }
            return result;
        }

        private float[] multiply(float[] input) {
            float[] result = new float[outputSize];
            for (int out = 0; out < outputSize; out++) {
                float sum = 0.0f;
                for (int in = 0; in < inputSize; in++) {
                    sum += weight[out][in] * input[in];
                }
                result[out] = sum;
            }
            return result;
        }
    }
}
