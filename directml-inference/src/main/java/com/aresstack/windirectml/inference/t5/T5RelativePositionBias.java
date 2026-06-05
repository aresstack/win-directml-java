package com.aresstack.windirectml.inference.t5;

/**
 * Computes T5 relative position buckets for the reference attention path.
 */
public final class T5RelativePositionBias {
    private final T5TensorData bias;
    private final int buckets;
    private final int maxDistance;
    private final int heads;

    public T5RelativePositionBias(T5TensorData bias, int buckets, int maxDistance, int heads) {
        this.bias = bias;
        this.buckets = Math.max(0, buckets);
        this.maxDistance = Math.max(1, maxDistance);
        this.heads = Math.max(1, heads);
    }

    public float value(int head, int queryPosition, int keyPosition, boolean bidirectional) {
        if (bias == null || buckets == 0) {
            return 0.0f;
        }
        int bucket = bucket(keyPosition - queryPosition, bidirectional);
        int safeHead = Math.floorMod(head, heads);
        if (bias.rank() == 2 && bucket < bias.dim(0) && safeHead < bias.dim(1)) {
            return bias.at(bucket, safeHead);
        }
        return 0.0f;
    }

    private int bucket(int relativePosition, boolean bidirectional) {
        int numBuckets = buckets;
        int ret = 0;
        int distance;
        if (bidirectional) {
            numBuckets /= 2;
            if (relativePosition > 0) {
                ret += numBuckets;
            }
            distance = Math.abs(relativePosition);
        } else {
            distance = Math.max(-relativePosition, 0);
        }
        int maxExact = Math.max(1, numBuckets / 2);
        if (distance < maxExact) {
            return Math.min(ret + distance, buckets - 1);
        }
        double logDistance = Math.log((double) distance / maxExact);
        double logMax = Math.log((double) maxDistance / maxExact);
        int large = maxExact + (int) (logDistance / logMax * (numBuckets - maxExact));
        large = Math.min(large, numBuckets - 1);
        return Math.min(ret + large, buckets - 1);
    }
}
