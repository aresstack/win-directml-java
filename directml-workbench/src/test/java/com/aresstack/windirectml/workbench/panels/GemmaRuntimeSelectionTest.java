package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.config.generation.GenerationModelRegistry;
import com.aresstack.windirectml.config.generation.GenerationOutputMode;
import com.aresstack.windirectml.inference.gemma.Gemma3RuntimeMode;
import com.aresstack.windirectml.runtime.facade.Backend;
import com.aresstack.windirectml.windows.WindowsBindings;
import com.aresstack.windirectml.workbench.WorkbenchModel;
import org.junit.jupiter.api.Test;

import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import java.lang.reflect.Field;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * WORKBENCH-CLEANUP-STREAMING-ONLY-1: the Gemma runtime selector and the visible "Show runtime profile"
 * checkbox are gone. Gemma's native/external choice comes from the general Backend (WARP → native), and
 * {@code -Dgemma.runtime} no longer drives the product/UI logic.
 */
class GemmaRuntimeSelectionTest {

    @Test
    void panelHasNoGemmaRuntimeSelectorOrVisibleProfileToggle() {
        for (Field f : SummarizerPanel.class.getDeclaredFields()) {
            String name = f.getName().toLowerCase();
            assertFalse(name.contains("gemmaruntime"),
                    "no Gemma runtime selector field should remain: " + f.getName());
            assertFalse(JComboBox.class.isAssignableFrom(f.getType()) && name.contains("runtime"),
                    "no Gemma runtime combo box should remain: " + f.getName());
            assertFalse(JCheckBox.class.isAssignableFrom(f.getType()) && name.contains("profile"),
                    "no visible profile checkbox should remain: " + f.getName());
        }
    }

    @Test
    void workbenchModelHasNoGemmaRuntimeState() {
        assertThrows(NoSuchMethodException.class,
                () -> WorkbenchModel.class.getMethod("getGemmaRuntimeMode"));
        assertThrows(NoSuchMethodException.class,
                () -> WorkbenchModel.class.getMethod("setGemmaRuntimeMode", Gemma3RuntimeMode.class));
    }

    @Test
    void gemmaUsesNativeDirectMlForWarpAndAuto() {
        // GEMMA-AUTO-GPU-1: WARP and AUTO both run the native DirectML runtime (WARP vs hardware adapter);
        // only CPU falls back to external Python.
        assertTrue(SummarizerPanel.gemmaUsesNativeDirectMl(Backend.WARP), "Backend=WARP -> native");
        assertTrue(SummarizerPanel.gemmaUsesNativeDirectMl(Backend.AUTO), "Backend=AUTO -> native (hardware)");
        assertFalse(SummarizerPanel.gemmaUsesNativeDirectMl(Backend.CPU), "Backend=CPU -> external");
        assertEquals(WindowsBindings.AdapterMode.WARP, SummarizerPanel.gemmaAdapterMode(Backend.WARP));
        assertEquals(WindowsBindings.AdapterMode.HARDWARE, SummarizerPanel.gemmaAdapterMode(Backend.AUTO));
    }

    @Test
    void gemmaRuntimePropertyDoesNotDriveTheDecision() {
        String prev = System.getProperty(Gemma3RuntimeMode.PROPERTY);
        System.setProperty(Gemma3RuntimeMode.PROPERTY, "native-warp"); // legacy flag set...
        try {
            // ...but the decision is purely backend-based and ignores it.
            assertFalse(SummarizerPanel.gemmaUsesNativeDirectMl(Backend.CPU),
                    "legacy -Dgemma.runtime must not force the native path for Backend=CPU");
            assertTrue(SummarizerPanel.gemmaUsesNativeDirectMl(Backend.WARP));
        } finally {
            if (prev == null) {
                System.clearProperty(Gemma3RuntimeMode.PROPERTY);
            } else {
                System.setProperty(Gemma3RuntimeMode.PROPERTY, prev);
            }
        }
    }

    @Test
    void gemma3ItIsSelectableAndRunnableAsAProduct() {
        // GEMMA-PRODUCT-1: the instruction-tuned Gemma 3 270M is a real, selectable, runnable product entry
        // (in the dropdown source = entries(), EXPERIMENTAL so the "planned/not executable" guard is skipped).
        GenerationModelRegistry.Entry e = GenerationModelRegistry.findByModelId("google/gemma-3-270m-it");
        assertTrue(e != null, "google/gemma-3-270m-it must be registered");
        assertTrue(GenerationModelRegistry.entries().contains(e), "must be in the summarizer dropdown source");
        assertTrue(e.isRunnable(), "must be runnable (not blocked by the PLANNED guard)");
        assertEquals(GenerationModelRegistry.Status.EXPERIMENTAL, e.status());
        assertEquals(GenerationModelRegistry.Architecture.CAUSAL_LM, e.architecture());
    }

    @Test
    void streamingIsTheDefaultOutputMode() {
        String prevOut = System.getProperty(GenerationOutputMode.OUTPUT_PROPERTY);
        String prevStream = System.getProperty(GenerationOutputMode.STREAMING_PROPERTY);
        System.clearProperty(GenerationOutputMode.OUTPUT_PROPERTY);
        System.clearProperty(GenerationOutputMode.STREAMING_PROPERTY);
        try {
            assertTrue(GenerationOutputMode.fromSystemProperty().isStreaming(),
                    "streaming output is the default");
        } finally {
            if (prevOut != null) {
                System.setProperty(GenerationOutputMode.OUTPUT_PROPERTY, prevOut);
            }
            if (prevStream != null) {
                System.setProperty(GenerationOutputMode.STREAMING_PROPERTY, prevStream);
            }
        }
    }
}
