package com.aresstack.windirectml.encoder.minilm;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Reiner Math-Test für die Bucket-Auswahl in
 * {@link DirectMlMiniLmEncoder#bucketFor(int)}. Keine GPU nötig.
 */
class DirectMlMiniLmEncoderBucketTest {

    @Test
    void exactBucketBoundariesMapToThemselves() {
        assertEquals(64, DirectMlMiniLmEncoder.bucketFor(64));
        assertEquals(128, DirectMlMiniLmEncoder.bucketFor(128));
        assertEquals(256, DirectMlMiniLmEncoder.bucketFor(256));
        assertEquals(512, DirectMlMiniLmEncoder.bucketFor(512));
    }

    @Test
    void shortSequencesRoundUpToSmallestBucket() {
        assertEquals(64, DirectMlMiniLmEncoder.bucketFor(1));
        assertEquals(64, DirectMlMiniLmEncoder.bucketFor(2));
        assertEquals(64, DirectMlMiniLmEncoder.bucketFor(63));
    }

    @Test
    void midSequencesRoundUpToNextBucket() {
        assertEquals(128, DirectMlMiniLmEncoder.bucketFor(65));
        assertEquals(128, DirectMlMiniLmEncoder.bucketFor(127));
        assertEquals(256, DirectMlMiniLmEncoder.bucketFor(129));
        assertEquals(256, DirectMlMiniLmEncoder.bucketFor(255));
        assertEquals(512, DirectMlMiniLmEncoder.bucketFor(257));
        assertEquals(512, DirectMlMiniLmEncoder.bucketFor(511));
    }

    @Test
    void overflowAboveLargestBucketReturnsExactLength() {
        // Past 512 we fall back to the exact length; no benefit from bucketing.
        assertEquals(513, DirectMlMiniLmEncoder.bucketFor(513));
        assertEquals(1024, DirectMlMiniLmEncoder.bucketFor(1024));
    }

    @Test
    void bucketsAreMonotonicallyIncreasing() {
        int[] b = DirectMlMiniLmEncoder.BUCKETS;
        for (int i = 1; i < b.length; i++) {
            assertTrue(b[i] > b[i - 1],
                    "BUCKETS must be strictly increasing at index " + i);
        }
    }
}

