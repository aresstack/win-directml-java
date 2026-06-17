package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.config.generation.GenerationModelRegistry;
import com.aresstack.windirectml.config.generation.GenerationModelRegistry.Entry;
import com.aresstack.windirectml.config.generation.GenerationModelRegistry.Status;
import com.aresstack.windirectml.inference.artifact.Gemma3PackageLifecycle;
import com.aresstack.windirectml.inference.artifact.ModelFamily;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * WORKBENCH-MODEL-STATUS-1: lock the generation-model status invariants the audit confirmed, so stale
 * planned/probe/experimental residue cannot creep back.
 *
 * <ul>
 *   <li>No registry note carries the old "Runtime integration is planned" / "until a native …" wording.</li>
 *   <li>Every model family with a working runtime is {@code runnable} (so the Summarizer's PLANNED
 *       "not executable yet" guard never blocks an executable model).</li>
 *   <li>The genuinely-not-executable models stay PLANNED (selectable, clearly marked).</li>
 *   <li>Gemma's Download-tab lifecycle is compiler-backed (Convert/READY), not download-only.</li>
 * </ul>
 */
class WorkbenchModelStatusAuditTest {

    @Test
    void noGenerationNoteHasStalePlannedIntegrationWording() {
        for (Entry e : GenerationModelRegistry.entries()) {
            String note = e.notes() == null ? "" : e.notes().toLowerCase();
            assertFalse(note.contains("runtime integration is planned"),
                    e.modelId() + " note has stale 'Runtime integration is planned' wording");
            assertFalse(note.contains("until a native java/warp"),
                    e.modelId() + " note has stale 'until a native Java/WARP …' wording");
        }
    }

    @Test
    void executableFamiliesAreRunnableSoTheyAreNotBlockedAsPlanned() {
        // These have a working Workbench runtime (native DirectML/WARP or wdmlpack-backed) -> must be runnable.
        for (String id : new String[]{
                "google/gemma-3-270m-it",
                "HuggingFaceTB/SmolLM2-135M-Instruct",
                "HuggingFaceTB/SmolLM2-360M-Instruct",
                "Salesforce/codet5-small",
                "Salesforce/codet5-base-multi-sum",
                "google-t5/t5-small",
                "google/flan-t5-small",
                "microsoft/Phi-3-mini-4k-instruct-onnx"}) {
            Entry e = GenerationModelRegistry.findByModelId(id);
            assertNotNull(e, id + " must be registered");
            assertTrue(e.isRunnable(), id + " must be runnable (SHIPPED/EXPERIMENTAL), not blocked as planned");
        }
    }

    @Test
    void genuinelyNotExecutableModelsStayPlanned() {
        // Selectable but clearly not executable yet -> PLANNED (the Summarizer shows the honest guard message).
        for (String id : new String[]{
                "microsoft/Phi-3.5-mini-instruct-onnx",
                "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                "Qwen/Qwen2.5-Coder-3B-Instruct",
                "google/gemma-3-270m"}) {
            Entry e = GenerationModelRegistry.findByModelId(id);
            assertNotNull(e, id + " must be registered");
            assertEquals(Status.PLANNED, e.status(), id + " should remain PLANNED (not executable yet)");
        }
    }

    @Test
    void gemmaDownloadTabLifecycleIsCompilerBacked() {
        // The Download tab must offer Convert/READY for Gemma (not "download-only / compiler missing").
        Gemma3PackageLifecycle lifecycle = new Gemma3PackageLifecycle();
        assertEquals(ModelFamily.GEMMA3, lifecycle.family());
        assertTrue(lifecycle.hasCompiler(), "Gemma must be compiler-backed (model_gemma3.wdmlpack)");
    }
}
