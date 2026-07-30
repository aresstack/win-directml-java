package com.aresstack.windirectml.config;

import com.aresstack.windirectml.catalog.CatalogModelFamily;
import com.aresstack.windirectml.catalog.LocalModelCatalog;
import com.aresstack.windirectml.catalog.LocalRuntimeModelDescriptor;
import com.aresstack.windirectml.catalog.ModelCapability;
import com.aresstack.windirectml.catalog.ModelStatus;
import com.aresstack.windirectml.config.generation.GenerationModelRegistry;

import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Drift guard across two <em>different</em> notions of "runnable":
 * <ul>
 *   <li>{@link GenerationModelRegistry} status = <b>workbench executability</b> (SHIPPED/EXPERIMENTAL =
 *       the workbench has a working runtime path for the model);</li>
 *   <li>{@link LocalModelCatalog} {@code RUNNABLE} = <b>release / AskAI certification</b> (a real green
 *       public-API package-only run, or a documented accepted exception).</li>
 * </ul>
 *
 * <p>The invariant is a <b>subset</b>, not equality: {@code catalog.runnable() ⊆ registry.executable()}.
 * A model may be workbench-executable while still {@code UNVERIFIED} in the catalog (e.g. Gemma, Phi-3),
 * but nothing may be catalog-{@code RUNNABLE} without a matching executable registry entry of the same
 * family/architecture. This keeps the guard strict in the direction that matters — the catalog can never
 * recommend a model the runtime cannot actually run — without forcing the two orthogonal statuses to be
 * identical.</p>
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
    void everyCatalogRunnableGenerationModelIsExecutableInTheRegistry() {
        // Subset invariant: catalog.runnable() ⊆ registry.executable(). The catalog may certify FEWER
        // models than the workbench can run, but never a model the registry cannot execute.
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
        assertTrue(lower(registryRunnable).containsAll(lower(catalogGeneration)),
                "a catalog-RUNNABLE generation model is not executable in GenerationModelRegistry: "
                        + "catalog=" + lower(catalogGeneration) + " registryExecutable=" + lower(registryRunnable));
    }

    @Test
    void gemmaAndPhi3AreTheDocumentedRegistryExecutableButCatalogUnverifiedDifference() {
        // The allowed, documented divergence between the two axes: the workbench can execute Gemma and
        // Phi-3 (registry runnable/EXPERIMENTAL), but they are NOT yet release-certified (catalog
        // UNVERIFIED). This asserts the difference explicitly so neither side drifts silently.
        for (String id : new String[]{"google/gemma-3-270m-it", "microsoft/Phi-3-mini-4k-instruct-onnx"}) {
            GenerationModelRegistry.Entry e = GenerationModelRegistry.findByModelId(id);
            assertNotNull(e, id + " must exist in the registry");
            assertTrue(e.isRunnable(), id + " is expected to be workbench-executable (registry runnable)");

            LocalRuntimeModelDescriptor d = LocalModelCatalog.findByRepositoryId(id);
            assertNotNull(d, id + " must exist in the catalog");
            assertEquals(ModelStatus.UNVERIFIED, d.status(), id + " must be catalog UNVERIFIED (not certified)");
            assertFalse(d.isRunnable(), id + " must not be catalog-runnable");
        }
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
