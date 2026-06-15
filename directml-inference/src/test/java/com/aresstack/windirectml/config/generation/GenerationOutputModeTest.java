package com.aresstack.windirectml.config.generation;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GENERATION-STREAMING-1: output-mode switch. Default is streaming; only an explicit buffered/false flips
 * it. (Lives in the inference test module because the config module's test task can't resolve
 * junit-platform-launcher offline; the production type stays in directml-config.)
 */
class GenerationOutputModeTest {

    @Test
    void defaultsToStreamingWhenUnset() {
        String prevOut = System.getProperty(GenerationOutputMode.OUTPUT_PROPERTY);
        String prevStream = System.getProperty(GenerationOutputMode.STREAMING_PROPERTY);
        System.clearProperty(GenerationOutputMode.OUTPUT_PROPERTY);
        System.clearProperty(GenerationOutputMode.STREAMING_PROPERTY);
        try {
            assertEquals(GenerationOutputMode.STREAMING, GenerationOutputMode.fromSystemProperty());
            assertTrue(GenerationOutputMode.STREAMING.isStreaming());
        } finally {
            restore(GenerationOutputMode.OUTPUT_PROPERTY, prevOut);
            restore(GenerationOutputMode.STREAMING_PROPERTY, prevStream);
        }
    }

    @Test
    void outputPropertyParsing() {
        assertEquals(GenerationOutputMode.STREAMING, GenerationOutputMode.fromOutputValue("streaming"));
        assertEquals(GenerationOutputMode.STREAMING, GenerationOutputMode.fromOutputValue("STREAMING"));
        assertEquals(GenerationOutputMode.STREAMING, GenerationOutputMode.fromOutputValue(""));
        assertEquals(GenerationOutputMode.STREAMING, GenerationOutputMode.fromOutputValue("weird"));
        assertEquals(GenerationOutputMode.BUFFERED, GenerationOutputMode.fromOutputValue("buffered"));
        assertEquals(GenerationOutputMode.BUFFERED, GenerationOutputMode.fromOutputValue("non-streaming"));
        assertEquals(GenerationOutputMode.BUFFERED, GenerationOutputMode.fromOutputValue("false"));
    }

    @Test
    void streamingBooleanParsing() {
        assertEquals(GenerationOutputMode.STREAMING, GenerationOutputMode.fromStreamingValue("true"));
        assertEquals(GenerationOutputMode.STREAMING, GenerationOutputMode.fromStreamingValue(""));
        assertEquals(GenerationOutputMode.STREAMING, GenerationOutputMode.fromStreamingValue("unknown"));
        assertEquals(GenerationOutputMode.BUFFERED, GenerationOutputMode.fromStreamingValue("false"));
        assertEquals(GenerationOutputMode.BUFFERED, GenerationOutputMode.fromStreamingValue("off"));
    }

    @Test
    void outputPropertyTakesPrecedenceOverStreamingProperty() {
        String prevOut = System.getProperty(GenerationOutputMode.OUTPUT_PROPERTY);
        String prevStream = System.getProperty(GenerationOutputMode.STREAMING_PROPERTY);
        System.setProperty(GenerationOutputMode.OUTPUT_PROPERTY, "buffered");
        System.setProperty(GenerationOutputMode.STREAMING_PROPERTY, "true");
        try {
            assertEquals(GenerationOutputMode.BUFFERED, GenerationOutputMode.fromSystemProperty());
        } finally {
            restore(GenerationOutputMode.OUTPUT_PROPERTY, prevOut);
            restore(GenerationOutputMode.STREAMING_PROPERTY, prevStream);
        }
    }

    private static void restore(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, value);
        }
    }
}
