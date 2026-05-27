package com.aresstack.windirectml.config.generation;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the shared generation API types.
 */
class GenerationApiTest {

    @Test
    void generationModelIdRejectsBlank() {
        assertThrows(IllegalArgumentException.class, () -> GenerationModelId.of(null));
        assertThrows(IllegalArgumentException.class, () -> GenerationModelId.of(""));
        assertThrows(IllegalArgumentException.class, () -> GenerationModelId.of("   "));
    }

    @Test
    void generationModelIdEquality() {
        GenerationModelId a = GenerationModelId.of("microsoft/Phi-3-mini");
        GenerationModelId b = GenerationModelId.of("microsoft/Phi-3-mini");
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
        assertEquals("microsoft/Phi-3-mini", a.value());
    }

    @Test
    void samplerConfigGreedy() {
        SamplerConfig greedy = SamplerConfig.greedy();
        assertTrue(greedy.isGreedy());
        assertEquals(0.0f, greedy.temperature());
        assertEquals(1, greedy.topK());
    }

    @Test
    void samplerConfigCustom() {
        SamplerConfig custom = SamplerConfig.of(0.7f, 40);
        assertFalse(custom.isGreedy());
        assertEquals(0.7f, custom.temperature());
        assertEquals(40, custom.topK());
    }

    @Test
    void samplerConfigRejectsInvalid() {
        assertThrows(IllegalArgumentException.class, () -> SamplerConfig.of(-1.0f, 1));
        assertThrows(IllegalArgumentException.class, () -> SamplerConfig.of(Float.NaN, 1));
        assertThrows(IllegalArgumentException.class, () -> SamplerConfig.of(Float.POSITIVE_INFINITY, 1));
        assertThrows(IllegalArgumentException.class, () -> SamplerConfig.of(0.5f, 0));
    }

    @Test
    void stopTokenPolicyEosOnly() {
        StopTokenPolicy policy = StopTokenPolicy.eosOnly();
        assertTrue(policy.stopStrings().isEmpty());
    }

    @Test
    void stopTokenPolicyWithStrings() {
        StopTokenPolicy policy = StopTokenPolicy.withStopStrings("<|end|>", "<|eot_id|>");
        assertEquals(2, policy.stopStrings().size());
        assertEquals("<|end|>", policy.stopStrings().get(0));
    }

    @Test
    void stopTokenPolicyRejectsNullOrBlankStrings() {
        assertSame(StopTokenPolicy.eosOnly(), StopTokenPolicy.withStopStrings((String[]) null));
        assertThrows(IllegalArgumentException.class, () -> StopTokenPolicy.withStopStrings("ok", null));
        assertThrows(IllegalArgumentException.class, () -> StopTokenPolicy.withStopStrings("ok", ""));
        assertThrows(IllegalArgumentException.class, () -> StopTokenPolicy.withStopStrings("ok", "   "));
    }

    @Test
    void generationRequestBuilder() {
        GenerationRequest req = GenerationRequest.builder()
                .userPrompt("Hello")
                .systemPrompt("You are helpful")
                .maxTokens(100)
                .sampler(SamplerConfig.greedy())
                .stopPolicy(StopTokenPolicy.eosOnly())
                .build();

        assertEquals("Hello", req.userPrompt());
        assertEquals("You are helpful", req.systemPrompt());
        assertEquals(100, req.maxTokens());
        assertTrue(req.sampler().isGreedy());
        assertTrue(req.stopPolicy().stopStrings().isEmpty());
    }

    @Test
    void generationRequestDefaults() {
        GenerationRequest req = GenerationRequest.builder().build();
        assertEquals("", req.userPrompt());
        assertEquals("", req.systemPrompt());
        assertEquals(0, req.maxTokens());
        assertTrue(req.sampler().isGreedy());
        assertNotNull(req.stopPolicy());
    }

    @Test
    void generationResult() {
        GenerationResult result = new GenerationResult("output text", "end_turn", 10, 5, 120);
        assertEquals("output text", result.text());
        assertEquals("end_turn", result.finishReason());
        assertEquals(10, result.promptTokens());
        assertEquals(5, result.completionTokens());
        assertEquals(15, result.totalTokens());
        assertEquals(120, result.elapsedMs());
    }

    @Test
    void generationResultNullDefaults() {
        GenerationResult result = new GenerationResult(null, null, 0, 0, 0);
        assertEquals("", result.text());
        assertEquals("unknown", result.finishReason());
    }

    @Test
    void generationExceptionChaining() {
        RuntimeException cause = new RuntimeException("oops");
        GenerationException ex = new GenerationException("failed", cause);
        assertEquals("failed", ex.getMessage());
        assertSame(cause, ex.getCause());
    }
}
