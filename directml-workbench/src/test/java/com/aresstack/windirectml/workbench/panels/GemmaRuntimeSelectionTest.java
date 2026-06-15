package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.inference.gemma.Gemma3RuntimeMode;
import com.aresstack.windirectml.workbench.WorkbenchModel;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * GEMMA-WORKBENCH-PROFILING-1: the Gemma runtime mode is UI-selectable via {@link WorkbenchModel} (no JVM
 * flag needed), external stays the default, and the selector labels are friendly.
 */
class GemmaRuntimeSelectionTest {

    @Test
    void defaultRuntimeIsExternalWhenFlagUnset() {
        String prev = System.getProperty(Gemma3RuntimeMode.PROPERTY);
        System.clearProperty(Gemma3RuntimeMode.PROPERTY);
        try {
            assertEquals(Gemma3RuntimeMode.EXTERNAL, new WorkbenchModel().getGemmaRuntimeMode());
        } finally {
            if (prev != null) {
                System.setProperty(Gemma3RuntimeMode.PROPERTY, prev);
            }
        }
    }

    @Test
    void uiSelectionDrivesRuntimeWithoutJvmFlag() {
        String prev = System.getProperty(Gemma3RuntimeMode.PROPERTY);
        System.clearProperty(Gemma3RuntimeMode.PROPERTY); // no -Dgemma.runtime
        try {
            WorkbenchModel model = new WorkbenchModel();
            assertEquals(Gemma3RuntimeMode.EXTERNAL, model.getGemmaRuntimeMode(), "starts external");
            // Selecting Native Java/WARP in the UI flips the authoritative state with no system property.
            model.setGemmaRuntimeMode(Gemma3RuntimeMode.NATIVE_WARP);
            assertEquals(Gemma3RuntimeMode.NATIVE_WARP, model.getGemmaRuntimeMode());
            // null falls back to external (defensive).
            model.setGemmaRuntimeMode(null);
            assertEquals(Gemma3RuntimeMode.EXTERNAL, model.getGemmaRuntimeMode());
        } finally {
            if (prev != null) {
                System.setProperty(Gemma3RuntimeMode.PROPERTY, prev);
            }
        }
    }

    @Test
    void legacyFlagSeedsInitialSelection() {
        String prev = System.getProperty(Gemma3RuntimeMode.PROPERTY);
        System.setProperty(Gemma3RuntimeMode.PROPERTY, "native-warp");
        try {
            assertEquals(Gemma3RuntimeMode.NATIVE_WARP, new WorkbenchModel().getGemmaRuntimeMode(),
                    "legacy -Dgemma.runtime=native-warp seeds the initial UI selection");
        } finally {
            if (prev == null) {
                System.clearProperty(Gemma3RuntimeMode.PROPERTY);
            } else {
                System.setProperty(Gemma3RuntimeMode.PROPERTY, prev);
            }
        }
    }

    @Test
    void selectorLabelsAreFriendly() {
        assertEquals("External Python / Transformers",
                SummarizerPanel.gemmaRuntimeLabel(Gemma3RuntimeMode.EXTERNAL));
        assertEquals("Native Java/WARP (experimental)",
                SummarizerPanel.gemmaRuntimeLabel(Gemma3RuntimeMode.NATIVE_WARP));
    }
}
