package com.aresstack.windirectml.inference.decoderonly;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Pure-Java behaviour of the shared decoder-only WARP KV cache. No device required; this guards the family-neutral
 * cache that Qwen-like and Llama-like (SmolLM2) runtimes share after the {@code decoderonly} extraction.
 */
class DecoderOnlyWarpKvCacheTest {

    // 2 KV heads x head dim 4 = keyWidth 8; 3 layers; max positions 16.
    private static DecoderOnlyConfig config() {
        return DecoderOnlyConfig.of(
                3,   // numHiddenLayers
                32,  // hiddenSize (unused here)
                64,  // intermediateSize (unused here)
                8,   // numAttentionHeads (unused here)
                2,   // numKeyValueHeads
                4,   // headDim
                16,  // maxPositionEmbeddings
                100, // vocabSize (unused here)
                1e-5f,
                10000.0f);
    }

    @Test
    void createsOneLayerCachePerHiddenLayerWithGqaKeyWidth() {
        DecoderOnlyWarpKvCache cache = DecoderOnlyWarpKvCache.create(config(), 10);

        assertEquals(3, cache.layerCount());
        for (int layer = 0; layer < cache.layerCount(); layer++) {
            assertEquals(8, cache.layer(layer).keyWidth());
            assertEquals(0, cache.layer(layer).size());
        }
        assertEquals(0, cache.completedTokenCount());
    }

    @Test
    void capacityIsBoundedByMaxPositionEmbeddings() {
        DecoderOnlyWarpKvCache cache = DecoderOnlyWarpKvCache.create(config(), 1000);
        DecoderOnlyWarpLayerKvCache layer = cache.layer(0);
        float[] key = new float[8];
        float[] value = new float[8];
        for (int i = 0; i < 16; i++) {
            layer.append(key, value);
        }
        // maxPositionEmbeddings == 16, so the 17th append must overflow even though maxTokens was 1000.
        assertThrows(IllegalStateException.class, () -> layer.append(key, value));
    }

    @Test
    void appendStoresKeysAndValuesContiguouslyAndAdvancesSize() {
        DecoderOnlyWarpLayerKvCache layer = new DecoderOnlyWarpLayerKvCache(2, 4);
        layer.append(new float[]{1f, 2f}, new float[]{10f, 20f});
        layer.append(new float[]{3f, 4f}, new float[]{30f, 40f});

        assertEquals(2, layer.size());
        assertArrayEquals(new float[]{1f, 2f, 3f, 4f, 0f, 0f, 0f, 0f}, layer.keys());
        assertArrayEquals(new float[]{10f, 20f, 30f, 40f, 0f, 0f, 0f, 0f}, layer.values());
    }

    @Test
    void appendFromOffsetSlicesSourceBuffers() {
        DecoderOnlyWarpLayerKvCache layer = new DecoderOnlyWarpLayerKvCache(2, 2);
        float[] keySource = {9f, 9f, 5f, 6f};   // append the slice at offset 2
        float[] valueSource = {9f, 9f, 7f, 8f};
        layer.append(keySource, 2, valueSource, 2);

        assertEquals(1, layer.size());
        assertArrayEquals(new float[]{5f, 6f, 0f, 0f}, layer.keys());
        assertArrayEquals(new float[]{7f, 8f, 0f, 0f}, layer.values());
    }

    @Test
    void completedTokenCountIsTheMinimumAcrossLayers() {
        DecoderOnlyWarpKvCache cache = DecoderOnlyWarpKvCache.create(config(), 10);
        float[] kv = new float[8];
        cache.layer(0).append(kv, kv);
        cache.layer(0).append(kv, kv);
        cache.layer(1).append(kv, kv);
        // layer 2 still empty → min is 0

        assertEquals(0, cache.completedTokenCount());

        cache.layer(1).append(kv, kv);
        cache.layer(2).append(kv, kv);
        cache.layer(2).append(kv, kv);
        // sizes are now [2, 2, 2]
        assertEquals(2, cache.completedTokenCount());
    }

    @Test
    void requireReadyForPositionRejectsLayersNotAtExpectedPosition() {
        DecoderOnlyWarpKvCache cache = DecoderOnlyWarpKvCache.create(config(), 10);
        float[] kv = new float[8];
        cache.layer(0).append(kv, kv);
        cache.layer(1).append(kv, kv);
        cache.layer(2).append(kv, kv);

        cache.requireReadyForPosition(1); // all layers at position 1 → ok
        assertThrows(IllegalStateException.class, () -> cache.requireReadyForPosition(2));
    }

    @Test
    void appendRejectsWidthMismatch() {
        DecoderOnlyWarpLayerKvCache layer = new DecoderOnlyWarpLayerKvCache(4, 2);
        assertThrows(IllegalArgumentException.class,
                () -> layer.append(new float[]{1f, 2f}, new float[]{1f, 2f}));
    }

    @Test
    void rejectsNonPositiveCapacityAndKeyWidth() {
        assertThrows(IllegalArgumentException.class, () -> new DecoderOnlyWarpLayerKvCache(0, 4));
        assertThrows(IllegalArgumentException.class, () -> new DecoderOnlyWarpLayerKvCache(4, 0));
        assertThrows(IllegalArgumentException.class, () -> DecoderOnlyWarpKvCache.create(config(), 0));
    }
}
