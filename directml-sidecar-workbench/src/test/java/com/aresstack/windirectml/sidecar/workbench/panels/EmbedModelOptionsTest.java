package com.aresstack.windirectml.sidecar.workbench.panels;

import com.aresstack.windirectml.config.models.EmbeddingModelRegistry;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies that the Workbench {@code embed.model} dropdown is built
 * from the shared {@link EmbeddingModelRegistry} and contains
 * <em>only</em> embedding entries. Decoder / summarizer entries from
 * the registry must never become selectable from the embedding
 * selector.
 */
class EmbedModelOptionsTest {

    @Test
    void containsBackwardCompatibleFamilyAliasesFirst() {
        List<String> opts = EmbedModelOptions.embeddingOptions();
        assertEquals("minilm", opts.get(0));
        assertEquals("e5", opts.get(1));
    }

    @Test
    void containsEveryEmbeddingModelIdFromTheRegistry() {
        List<String> opts = EmbedModelOptions.embeddingOptions();
        for (EmbeddingModelRegistry.Entry e : EmbeddingModelRegistry
                .entriesByUseCase(EmbeddingModelRegistry.UseCase.EMBEDDING)) {
            assertTrue(opts.contains(e.modelId()),
                    "embedding modelId missing from workbench dropdown: " + e.modelId());
        }
    }

    @Test
    void decoderAndSummarizerModelsAreNotSelectable() {
        List<String> opts = EmbedModelOptions.embeddingOptions();
        // The three known non-embedding registry IDs.
        assertFalse(opts.contains("openai/gpt-oss-120b"),
                "decoder model must not appear in embedding dropdown");
        assertFalse(opts.contains("casperhansen/llama-3.3-70b-instruct-awq"),
                "decoder model must not appear in embedding dropdown");
        assertFalse(opts.contains("ellamind/summarizer-v6-llama-v2"),
                "summarizer model must not appear in embedding dropdown");
    }

    @Test
    void everyOptionIsEitherAFamilyAliasOrAnEmbeddingRegistryEntry() {
        List<String> opts = EmbedModelOptions.embeddingOptions();
        for (String opt : opts) {
            if ("minilm".equals(opt) || "e5".equals(opt)) {
                continue;
            }
            EmbeddingModelRegistry.Entry entry =
                    EmbeddingModelRegistry.findByModelId(opt);
            assertTrue(entry != null,
                    "dropdown option not backed by a registry entry: " + opt);
            assertEquals(EmbeddingModelRegistry.UseCase.EMBEDDING, entry.useCase(),
                    "non-embedding option leaked into dropdown: " + opt);
        }
    }
}
