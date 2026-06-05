package com.aresstack.windirectml.inference.smollm2;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SmolLM2ConfigTest {

    @Test
    void supports135mStyleConfig() {
        SmolLM2Config config = SmolLM2TestFixtures.config135(true);
        SmolLM2ModelFamily family = new SmolLM2ModelFamily();

        assertTrue(family.supports(config));
        SmolLM2Architecture architecture = family.architecture(config);
        assertEquals("smollm2", family.id());
        assertEquals("SmolLM2", family.displayName());
        assertEquals(8, architecture.hiddenSize());
        assertTrue(architecture.usesGroupedQueryAttention());
    }

    @Test
    void supports360mStyleConfig() {
        SmolLM2Config config = SmolLM2TestFixtures.config360(true);
        SmolLM2Architecture architecture = new SmolLM2ModelFamily().architecture(config);

        assertEquals(3, architecture.keyValueHeads());
        assertTrue(architecture.usesGroupedQueryAttention());
    }

    @Test
    void rejectsUnsupportedModelType() {
        SmolLM2Config config = new SmolLM2Config("qwen2", java.util.List.of("LlamaForCausalLM"),
                8, 16, 1, 4, 2, 2, 32, 64, 1.0e-5d, 10000.0d,
                "silu", false, false, 1, 2, null, true);

        assertFalse(new SmolLM2ModelFamily().supports(config));
    }
}
