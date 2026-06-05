package com.aresstack.windirectml.inference.t5;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class T5ModelFamilyTest {

    @Test
    void supportsValidEncoderDecoderT5Config() {
        assertTrue(new T5ModelFamily().supports(T5TestFixtures.tinyConfig(false)));
    }

    @Test
    void rejectsDecoderOnlyConfig() {
        T5Config config = new T5Config(List.of("T5ForConditionalGeneration"), "t5", false,
                4, 2, 8, 1, 1, 2, 6, 4, 16, 1e-6f,
                0, 2, 0, true, "relu");

        assertFalse(new T5ModelFamily().supports(config));
    }

    @Test
    void createsArchitectureFromConfig() {
        T5Architecture architecture = new T5ModelFamily().architecture(T5TestFixtures.tinyConfig(true));

        assertEquals(4, architecture.modelSize());
        assertEquals(4, architecture.attentionInnerSize());
        assertEquals(1, architecture.encoderLayers());
        assertEquals(1, architecture.decoderLayers());
        assertTrue(architecture.usesGatedFeedForward());
    }

    @Test
    void createsSpecialTokensFromConfig() {
        T5SpecialTokens tokens = new T5ModelFamily().specialTokens(T5TestFixtures.tinyConfig(false));

        assertEquals(0, tokens.padTokenId());
        assertEquals(2, tokens.eosTokenId());
        assertEquals(0, tokens.decoderStartTokenId());
    }

    @Test
    void packageMetadataExportsStableManifestValues() {
        T5PackageMetadata metadata = new T5ModelFamily().packageMetadata(T5TestFixtures.tinyConfig(false));

        Map<String, Object> manifest = metadata.toManifest();

        assertEquals("t5", manifest.get("modelFamily"));
        assertEquals("encoder-decoder", manifest.get("architecture"));
        assertEquals(4, manifest.get("dModel"));
        assertEquals(2, manifest.get("dKv"));
        assertEquals(8, manifest.get("dFf"));
        assertEquals("relu", manifest.get("feedForwardProjection"));
        assertEquals(true, manifest.get("tieWordEmbeddings"));
    }
}
