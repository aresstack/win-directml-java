package com.aresstack.windirectml.config.generation;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Locale;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link GenerationModelRegistry}.
 */
class GenerationModelRegistryTest {

    @Test
    void registryContainsPhi3Experimental() {
        GenerationModelRegistry.Entry phi3 =
                GenerationModelRegistry.findByModelId("microsoft/Phi-3-mini-4k-instruct-onnx");
        assertNotNull(phi3, "Phi-3 must be in the generation registry");
        assertEquals(GenerationModelRegistry.Architecture.CAUSAL_LM, phi3.architecture());
        assertEquals(GenerationModelRegistry.Status.EXPERIMENTAL, phi3.status());
        assertEquals(ChatTemplate.PHI3, phi3.chatTemplate());
        assertTrue(phi3.isCausalLM());
        assertTrue(phi3.isRunnable());
    }

    @Test
    void registryContainsQwenModelsAsPlanned() {
        String[] qwenIds = {
                "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                "Qwen/Qwen2.5-Coder-3B-Instruct"
        };
        for (String id : qwenIds) {
            GenerationModelRegistry.Entry entry = GenerationModelRegistry.findByModelId(id);
            assertNotNull(entry, "registry must contain " + id);
            assertEquals(GenerationModelRegistry.Status.PLANNED, entry.status(),
                    id + " should be planned");
            assertEquals(ChatTemplate.CHATML, entry.chatTemplate());
            assertTrue(entry.isCausalLM());
            assertFalse(entry.isRunnable(),
                    id + " must NOT be advertised as runnable");
        }
    }


    @Test
    void registryContainsSmolLm2ModelsAsExperimental() {
        String[] smolIds = {
                "HuggingFaceTB/SmolLM2-135M-Instruct",
                "HuggingFaceTB/SmolLM2-360M-Instruct"
        };
        for (String id : smolIds) {
            GenerationModelRegistry.Entry entry = GenerationModelRegistry.findByModelId(id);
            assertNotNull(entry, "registry must contain " + id);
            assertEquals(GenerationModelRegistry.Architecture.CAUSAL_LM, entry.architecture());
            assertEquals(GenerationModelRegistry.Status.EXPERIMENTAL, entry.status());
            assertTrue(entry.isRunnable(), id + " should be advertised as reference-runtime runnable");
            assertEquals(ChatTemplate.RAW, entry.chatTemplate());
        }
    }

    @Test
    void registryContainsT5FamilyAsExperimentalSeq2Seq() {
        String[] ids = {"Salesforce/codet5-small", "google-t5/t5-small", "google/flan-t5-small"};

        for (String id : ids) {
            GenerationModelRegistry.Entry entry = GenerationModelRegistry.findByModelId(id);

            assertNotNull(entry, "registry must contain " + id);
            assertEquals(GenerationModelRegistry.Architecture.SEQ2SEQ, entry.architecture());
            assertEquals(GenerationModelRegistry.Status.EXPERIMENTAL, entry.status());
            assertEquals(ChatTemplate.RAW, entry.chatTemplate());
            assertTrue(entry.isRunnable());
        }
    }

    @Test
    void noQwenModelIsRunnable() {
        List<GenerationModelRegistry.Entry> runnable = GenerationModelRegistry.runnableEntries();
        for (GenerationModelRegistry.Entry e : runnable) {
            assertFalse(e.modelId().toLowerCase(Locale.ROOT).contains("qwen"),
                    "No Qwen model should be runnable yet: " + e.modelId());
        }
    }

    @Test
    void runnableEntriesOnlyContainsExperimentalOrShipped() {
        List<GenerationModelRegistry.Entry> runnable = GenerationModelRegistry.runnableEntries();
        assertFalse(runnable.isEmpty(), "at least Phi-3 should be runnable");
        for (GenerationModelRegistry.Entry e : runnable) {
            assertTrue(e.status() == GenerationModelRegistry.Status.SHIPPED
                            || e.status() == GenerationModelRegistry.Status.EXPERIMENTAL,
                    e.modelId() + " has unexpected status: " + e.status());
        }
    }

    @Test
    void findByModelIdIsCaseInsensitive() {
        GenerationModelRegistry.Entry e1 =
                GenerationModelRegistry.findByModelId("MICROSOFT/PHI-3-MINI-4K-INSTRUCT-ONNX");
        assertNotNull(e1);
        assertEquals("microsoft/Phi-3-mini-4k-instruct-onnx", e1.modelId());
    }

    @Test
    void findByModelIdReturnsNullForUnknown() {
        assertNull(GenerationModelRegistry.findByModelId("unknown/model"));
        assertNull(GenerationModelRegistry.findByModelId(null));
        assertNull(GenerationModelRegistry.findByModelId(""));
        assertNull(GenerationModelRegistry.findByModelId("   "));
    }

    @Test
    void entriesByArchitectureFiltersCorrectly() {
        List<GenerationModelRegistry.Entry> causal =
                GenerationModelRegistry.entriesByArchitecture(GenerationModelRegistry.Architecture.CAUSAL_LM);
        assertFalse(causal.isEmpty());
        for (GenerationModelRegistry.Entry e : causal) {
            assertEquals(GenerationModelRegistry.Architecture.CAUSAL_LM, e.architecture());
        }

        List<GenerationModelRegistry.Entry> seq2seq =
                GenerationModelRegistry.entriesByArchitecture(GenerationModelRegistry.Architecture.SEQ2SEQ);
        assertFalse(seq2seq.isEmpty(), "T5/CodeT5 family should now be visible");
    }

    @Test
    void entriesByStatusFiltersCorrectly() {
        List<GenerationModelRegistry.Entry> planned =
                GenerationModelRegistry.entriesByStatus(GenerationModelRegistry.Status.PLANNED);
        assertTrue(planned.size() >= 3, "at least Phi-3.5 + 3 Qwen should be planned");
        for (GenerationModelRegistry.Entry e : planned) {
            assertEquals(GenerationModelRegistry.Status.PLANNED, e.status());
        }
    }

    @Test
    void allEntriesHaveNonBlankModelId() {
        for (GenerationModelRegistry.Entry e : GenerationModelRegistry.entries()) {
            assertNotNull(e.modelId());
            assertFalse(e.modelId().trim().isEmpty());
        }
    }

    @Test
    void allEntriesHaveModelDirHints() {
        for (GenerationModelRegistry.Entry e : GenerationModelRegistry.entries()) {
            assertNotNull(e.modelDirHints());
            assertFalse(e.modelDirHints().isEmpty(),
                    e.modelId() + " should have at least one model dir hint");
        }
    }

    @Test
    void entryTrimsModelIdForConsistentLookupKeys() {
        GenerationModelRegistry.Entry entry = new GenerationModelRegistry.Entry(
                "  acme/model  ",
                GenerationModelRegistry.Architecture.CAUSAL_LM,
                "Acme",
                "1B",
                ChatTemplate.RAW,
                GenerationModelRegistry.Status.PLANNED,
                java.util.Collections.singletonList("model/acme/model"),
                null
        );
        assertEquals("acme/model", entry.modelId());
    }
}
