package com.aresstack.windirectml.inference.t5;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class T5TensorNameMapperTest {

    @Test
    void mapsEncoderAndDecoderAttentionRoles() {
        T5Config config = T5TestFixtures.tinyConfig(false);

        T5ExpectedTensor encoderQ = T5TensorNameMapper.encoderSelfAttention(0, "q", config);
        T5ExpectedTensor decoderCrossQ = T5TensorNameMapper.decoderCrossAttention(0, "q", config);

        assertEquals("encoder.layer.0.self_attn.q", encoderQ.role());
        assertEquals("encoder.layers.000.self_attn.q.weight", encoderQ.runtimeName());
        assertArrayEquals(new long[]{config.attentionInnerSize(), config.modelSize()}, encoderQ.expectedDims());
        assertEquals("decoder.layer.0.cross_attn.q", decoderCrossQ.role());
        assertEquals("decoder.layers.000.cross_attn.q.weight", decoderCrossQ.runtimeName());
    }

    @Test
    void mapsGatedFeedForwardRoles() {
        T5Config config = T5TestFixtures.tinyConfig(true);

        T5ExpectedTensor wi0 = T5TensorNameMapper.encoderFeedForward(0, "wi_0", config);
        T5ExpectedTensor wi1 = T5TensorNameMapper.decoderFeedForward(0, "wi_1", config);

        assertEquals("encoder.layer.0.ffn.wi_0", wi0.role());
        assertEquals("encoder.block.0.layer.1.DenseReluDense.wi_0.weight", wi0.sourceName());
        assertArrayEquals(new long[]{config.feedForwardSize(), config.modelSize()}, wi0.expectedDims());
        assertEquals("decoder.layer.0.ffn.wi_1", wi1.role());
    }
}
