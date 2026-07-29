package com.aresstack.windirectml.config;

import com.aresstack.windirectml.catalog.CatalogModelFamily;
import com.aresstack.windirectml.catalog.LocalModelCatalog;
import com.aresstack.windirectml.catalog.LocalRuntimeModelDescriptor;
import com.aresstack.windirectml.catalog.ModelCapability;
import com.aresstack.windirectml.config.generation.GenerationModelRegistry;

import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Strict drift guard: the neutral {@link LocalModelCatalog} and the legacy {@link GenerationModelRegistry}
 * must agree on which generation checkpoints are runnable and on their architecture. If either list changes
 * without the other, this fails — so the catalog stays the single source of truth and the registries never
 * silently diverge from it (or from the workbench that also consumes the registry).
 */
class CatalogRegistryParityTest {

    private static Set<String> lower(Set<String> in) {
        Set<String> out = new HashSet<String>();
        for (String s : in) {
            out.add(s.toLowerCase(Locale.ROOT));
        }
        return out;
    }

    private static boolean isGenerationFamily(CatalogModelFamily f) {
        return f == CatalogModelFamily.QWEN || f == CatalogModelFamily.SMOLLM2
                || f == CatalogModelFamily.GEMMA3 || f == CatalogModelFamily.PHI3
                || f == CatalogModelFamily.T5;
    }

    @Test
    void runnableGenerationSetMatchesTheLegacyRegistry() {
        Set<String> catalogGeneration = new HashSet<String>();
        for (LocalRuntimeModelDescriptor d : LocalModelCatalog.runnable()) {
            if (isGenerationFamily(d.runtimeFamily())) {
                catalogGeneration.add(d.huggingFaceRepositoryId());
            }
        }
        Set<String> registryRunnable = new HashSet<String>();
        for (GenerationModelRegistry.Entry e : GenerationModelRegistry.runnableEntries()) {
            registryRunnable.add(e.modelId());
        }
        assertEquals(lower(registryRunnable), lower(catalogGeneration),
                "catalog runnable generation set drifted from GenerationModelRegistry.runnableEntries()");
    }

    @Test
    void architecturesAgreeForEveryRunnableGenerationModel() {
        for (LocalRuntimeModelDescriptor d : LocalModelCatalog.runnable()) {
            if (!isGenerationFamily(d.runtimeFamily())) {
                continue;
            }
            GenerationModelRegistry.Entry e = GenerationModelRegistry.findByModelId(d.huggingFaceRepositoryId());
            assertNotNull(e, d.huggingFaceRepositoryId() + " missing from GenerationModelRegistry");
            boolean catalogCausal = d.hasCapability(ModelCapability.CHAT)
                    && d.runtimeFamily() != CatalogModelFamily.T5;
            boolean catalogSeq2Seq = d.hasCapability(ModelCapability.SEQ2SEQ);
            if (e.architecture() == GenerationModelRegistry.Architecture.SEQ2SEQ) {
                assertEquals(true, catalogSeq2Seq, d.huggingFaceRepositoryId() + " should be seq2seq");
            } else {
                assertEquals(true, catalogCausal, d.huggingFaceRepositoryId() + " should be causal-lm");
            }
        }
    }

    @Test
    void catalogGenerationModelsAreNeverClassifiedAsEmbeddings() {
        // A generation checkpoint must not also be reachable as an embedding capability in the catalog.
        List<LocalRuntimeModelDescriptor> embeddings =
                LocalModelCatalog.runnableByCapability(ModelCapability.EMBEDDING);
        Set<String> embeddingRepos = new HashSet<String>();
        for (LocalRuntimeModelDescriptor d : embeddings) {
            embeddingRepos.add(d.huggingFaceRepositoryId().toLowerCase(Locale.ROOT));
        }
        for (LocalRuntimeModelDescriptor d : LocalModelCatalog.runnable()) {
            if (isGenerationFamily(d.runtimeFamily())) {
                assertFalse(embeddingRepos.contains(d.huggingFaceRepositoryId().toLowerCase(Locale.ROOT)),
                        d.huggingFaceRepositoryId() + " must not be an embedding");
            }
        }
    }
}
