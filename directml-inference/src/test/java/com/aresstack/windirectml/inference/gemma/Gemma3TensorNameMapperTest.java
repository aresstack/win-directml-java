package com.aresstack.windirectml.inference.gemma;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** GEMMA-WARP-1: tensor name/shape mapping incl. Gemma-specific QK-norm + sandwich norms, tied LM head. */
class Gemma3TensorNameMapperTest {

    private static Gemma3Config tinyConfig() {
        return new Gemma3Config("gemma3_text", List.of("Gemma3ForCausalLM"),
                640, 2048, 2, 4, 1, 256, 262144, 32768,
                1e-6, 1_000_000, 10_000, 512, 6, List.of(), 256, "gelu_pytorch_tanh", 2, 1, 0, true);
    }

    @Test
    void requiredNamesCoverEveryLayerTensorAndAreUnique() {
        Gemma3Config c = tinyConfig();
        List<String> names = Gemma3TensorNameMapper.requiredTensorNames(c);
        // embed + final norm + 13 tensors per layer.
        assertEquals(2 + 13 * c.numHiddenLayers(), names.size());
        assertEquals(names.size(), names.stream().distinct().count(), "names must be unique");
        assertTrue(names.contains(Gemma3TensorNameMapper.EMBED_TOKENS));
        assertTrue(names.contains(Gemma3TensorNameMapper.FINAL_NORM));
        // Gemma-specific tensors present.
        assertTrue(names.contains(Gemma3TensorNameMapper.qNorm(0)));
        assertTrue(names.contains(Gemma3TensorNameMapper.kNorm(0)));
        assertTrue(names.contains(Gemma3TensorNameMapper.preFeedforwardLayerNorm(1)));
        assertTrue(names.contains(Gemma3TensorNameMapper.postFeedforwardLayerNorm(1)));
        // No separate lm_head (tied).
        assertFalse(names.stream().anyMatch(n -> n.contains("lm_head")));
    }

    @Test
    void expectedShapesUseDecoupledHeadDim() {
        Gemma3Config c = tinyConfig();
        Map<String, long[]> shapes = Gemma3TensorNameMapper.expectedShapes(c);
        assertArrayEquals(new long[]{262144, 640}, shapes.get(Gemma3TensorNameMapper.EMBED_TOKENS));
        assertArrayEquals(new long[]{1024, 640}, shapes.get(Gemma3TensorNameMapper.qProj(0)));  // attn=4*256
        assertArrayEquals(new long[]{256, 640}, shapes.get(Gemma3TensorNameMapper.kProj(0)));   // kv=1*256
        assertArrayEquals(new long[]{640, 1024}, shapes.get(Gemma3TensorNameMapper.oProj(0)));
        assertArrayEquals(new long[]{256}, shapes.get(Gemma3TensorNameMapper.qNorm(0)));        // head_dim
        assertArrayEquals(new long[]{2048, 640}, shapes.get(Gemma3TensorNameMapper.gateProj(0)));
        assertArrayEquals(new long[]{640, 2048}, shapes.get(Gemma3TensorNameMapper.downProj(0)));
    }
}
