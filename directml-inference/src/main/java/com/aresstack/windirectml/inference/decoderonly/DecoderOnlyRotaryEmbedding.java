package com.aresstack.windirectml.inference.decoderonly;

/**
 * Applies decoder-only rotary position embeddings to query/key vectors.
 */
public final class DecoderOnlyRotaryEmbedding {

    private final int headDim;
    private final double theta;
    private final float[] cos;
    private final float[] sin;

    public DecoderOnlyRotaryEmbedding(int headDim, double theta, int maxPrecomputedPositions) {
        if (headDim <= 0 || headDim % 2 != 0) {
            throw new IllegalArgumentException("headDim must be positive and even");
        }
        if (maxPrecomputedPositions < 0) {
            throw new IllegalArgumentException("maxPrecomputedPositions must not be negative");
        }
        this.headDim = headDim;
        this.theta = theta;
        int halfDim = headDim / 2;
        this.cos = new float[maxPrecomputedPositions * halfDim];
        this.sin = new float[maxPrecomputedPositions * halfDim];
        for (int pos = 0; pos < maxPrecomputedPositions; pos++) {
            for (int i = 0; i < halfDim; i++) {
                double angle = positionAngle(pos, i);
                cos[pos * halfDim + i] = (float) Math.cos(angle);
                sin[pos * halfDim + i] = (float) Math.sin(angle);
            }
        }
    }

    public int headDim() {
        return headDim;
    }

    public int precomputedPositions() {
        return cos.length / (headDim / 2);
    }

    public void apply(float[] vector, int offset, int position) {
        int halfDim = headDim / 2;
        int ropeMaxPos = precomputedPositions();
        if (position < ropeMaxPos) {
            for (int i = 0; i < halfDim; i++) {
                rotate(vector, offset, i, cos[position * halfDim + i], sin[position * halfDim + i]);
            }
            return;
        }
        for (int i = 0; i < halfDim; i++) {
            double angle = positionAngle(position, i);
            rotate(vector, offset, i, (float) Math.cos(angle), (float) Math.sin(angle));
        }
    }

    private double positionAngle(int position, int halfDimOffset) {
        return position * (1.0 / Math.pow(theta, (2.0 * halfDimOffset) / headDim));
    }

    private void rotate(float[] vector, int offset, int halfDimOffset, float cosValue, float sinValue) {
        int halfDim = headDim / 2;
        float x0 = vector[offset + halfDimOffset];
        float x1 = vector[offset + halfDim + halfDimOffset];
        vector[offset + halfDimOffset] = x0 * cosValue - x1 * sinValue;
        vector[offset + halfDim + halfDimOffset] = x0 * sinValue + x1 * cosValue;
    }
}
