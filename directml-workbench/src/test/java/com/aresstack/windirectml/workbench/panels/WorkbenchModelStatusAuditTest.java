package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.config.generation.GenerationModelRegistry;
import com.aresstack.windirectml.config.generation.GenerationModelRegistry.Entry;
import com.aresstack.windirectml.config.generation.GenerationModelRegistry.Status;
import com.aresstack.windirectml.inference.artifact.Gemma3PackageLifecycle;
import com.aresstack.windirectml.inference.artifact.ModelFamily;
import org.junit.jupiter.api.Test;

import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Collectors;

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
                "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                "HuggingFaceTB/SmolLM2-135M-Instruct",
                "HuggingFaceTB/SmolLM2-360M-Instruct",
                "Salesforce/codet5-small",
                "Salesforce/codet5-base-multi-sum",
                "google-t5/t5-small",
                "google/flan-t5-small"}) {
            Entry e = GenerationModelRegistry.findByModelId(id);
            assertNotNull(e, id + " must be registered");
            assertTrue(e.isRunnable(), id + " must be runnable (SHIPPED/EXPERIMENTAL), not blocked as planned");
        }
    }

    @Test
    void qwen05bIsRunnableByStatusNotByException() {
        // WORKBENCH-MODEL-STATUS-2: Qwen 0.5B is executable, so it is EXPERIMENTAL (runnable) by status and no
        // longer needs the qwenTestModel PLANNED-guard exemption to run.
        Entry e = GenerationModelRegistry.findByModelId("Qwen/Qwen2.5-Coder-0.5B-Instruct");
        assertNotNull(e);
        assertEquals(Status.EXPERIMENTAL, e.status(), "Qwen 0.5B is runnable -> EXPERIMENTAL, not PLANNED");
        assertTrue(e.isRunnable());
    }

    @Test
    void smolLm2IsRunnableByStatusWithHonestNotes() {
        // SMOLLM2-PRODUCT-AUDIT-1: both SmolLM2 variants are EXPERIMENTAL/runnable (no PLANNED-guard exemption
        // needed), and their notes carry no stale planned/probe/not-implemented wording.
        for (String id : new String[]{"HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct"}) {
            Entry e = GenerationModelRegistry.findByModelId(id);
            assertNotNull(e, id + " must be registered");
            assertEquals(Status.EXPERIMENTAL, e.status(), id + " is runnable -> EXPERIMENTAL");
            assertTrue(e.isRunnable());
            String note = e.notes() == null ? "" : e.notes().toLowerCase();
            for (String banned : new String[]{"planned", "not implemented", "probe", "runtime integration"}) {
                assertFalse(note.contains(banned), id + " note has stale wording '" + banned + "': " + note);
            }
        }
    }

    @Test
    void t5FamilyIsRunnableByStatusWithHonestNotes() {
        // T5-PRODUCT-AUDIT-1: every curated T5/Flan-T5/CodeT5 model is EXPERIMENTAL/runnable (no PLANNED-guard
        // exemption needed -- they are not PLANNED), and the registry notes carry no stale planned/probe/
        // not-implemented wording. Internal "experimental" wording is allowed; "runtime integration is planned"
        // is not.
        for (String id : new String[]{
                "Salesforce/codet5-small",
                "Salesforce/codet5-base-multi-sum",
                "google-t5/t5-small",
                "google/flan-t5-small"}) {
            Entry e = GenerationModelRegistry.findByModelId(id);
            assertNotNull(e, id + " must be registered");
            assertEquals(Status.EXPERIMENTAL, e.status(), id + " is runnable -> EXPERIMENTAL");
            assertTrue(e.isRunnable());
            String note = e.notes() == null ? "" : e.notes().toLowerCase();
            for (String banned : new String[]{"planned", "not implemented", "probe", "runtime integration"}) {
                assertFalse(note.contains(banned), id + " note has stale wording '" + banned + "': " + note);
            }
        }
    }

    @Test
    void genuinelyNotExecutableModelsStayPlanned() {
        // Selectable but clearly not executable yet -> PLANNED (the Summarizer shows the honest guard message).
        for (String id : new String[]{
                "microsoft/Phi-3-mini-4k-instruct-onnx",
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
    void phiModelsAreNotRunnableAndCarryHonestNotes() {
        // PHI3-PRODUCT-AUDIT-1: both Phi-3 models are selectable + downloadable but NOT executable in the
        // Workbench (the homogeneous artifact gate has no wdmlpack compiler for Phi-3). They must be PLANNED
        // (so the Summarizer guard blocks them upfront) and must not claim ONNX-Runtime/GenAI execution.
        for (String id : new String[]{
                "microsoft/Phi-3-mini-4k-instruct-onnx",
                "microsoft/Phi-3.5-mini-instruct-onnx"}) {
            Entry e = GenerationModelRegistry.findByModelId(id);
            assertNotNull(e, id + " must be registered");
            assertEquals(Status.PLANNED, e.status(), id + " is not executable in the Workbench -> PLANNED");
            assertFalse(e.isRunnable(), id + " must not be runnable by status");
            String note = e.notes() == null ? "" : e.notes().toLowerCase();
            // Ban the false positive claim (ONNX Runtime / GenAI execution); honest negations like
            // "no Python/ONNX Runtime" are fine.
            assertFalse(note.contains("uses onnx runtime"), id + " note must not claim ONNX Runtime execution: " + note);
            assertFalse(note.contains("genai"), id + " note must not claim an ONNX Runtime GenAI path: " + note);
            assertFalse(note.contains("first supported"),
                    id + " note must not claim it is the first supported runnable backend: " + note);
        }
    }

    @Test
    void runnableSetIsExactlyTheKnownProductModels() {
        // WORKBENCH-MODEL-STATUS-CLOSEOUT-1: pin the complete runnable (SHIPPED/EXPERIMENTAL) set so a new model
        // cannot silently become "runnable by status" without an explicit audit. Every id here has a verified
        // executable Workbench path; Phi-3 is intentionally absent (gate-blocked -> PLANNED).
        Set<String> expected = new TreeSet<>(Set.of(
                "google/gemma-3-270m-it",
                "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                "HuggingFaceTB/SmolLM2-135M-Instruct",
                "HuggingFaceTB/SmolLM2-360M-Instruct",
                "Salesforce/codet5-small",
                "Salesforce/codet5-base-multi-sum",
                "google-t5/t5-small",
                "google/flan-t5-small"));
        Set<String> actual = GenerationModelRegistry.entries().stream()
                .filter(Entry::isRunnable)
                .map(Entry::modelId)
                .collect(Collectors.toCollection(TreeSet::new));
        assertEquals(expected, actual,
                "Runnable-by-status set changed; audit the new/removed model before updating this lock-in");
    }

    @Test
    void gemmaDownloadTabLifecycleIsCompilerBacked() {
        // The Download tab must offer Convert/READY for Gemma (not "download-only / compiler missing").
        Gemma3PackageLifecycle lifecycle = new Gemma3PackageLifecycle();
        assertEquals(ModelFamily.GEMMA3, lifecycle.family());
        assertTrue(lifecycle.hasCompiler(), "Gemma must be compiler-backed (model_gemma3.wdmlpack)");
    }
}
