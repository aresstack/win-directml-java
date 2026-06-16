package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.config.generation.GenerationOutputMode;
import com.aresstack.windirectml.inference.gemma.Gemma3RuntimeMode;
import com.aresstack.windirectml.runtime.facade.Backend;
import com.aresstack.windirectml.workbench.WorkbenchModel;
import org.junit.jupiter.api.Test;

import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import java.lang.reflect.Field;

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
    void gemmaUsesNativeWarpWhenBackendIsWarp() {
        assertTrue(SummarizerPanel.gemmaUsesNativeWarp(Backend.WARP), "Backend=WARP -> native Java/WARP");
        assertFalse(SummarizerPanel.gemmaUsesNativeWarp(Backend.AUTO), "Backend=AUTO -> external");
        assertFalse(SummarizerPanel.gemmaUsesNativeWarp(Backend.CPU), "Backend=CPU -> external");
    }

    @Test
    void gemmaRuntimePropertyDoesNotDriveTheDecision() {
        String prev = System.getProperty(Gemma3RuntimeMode.PROPERTY);
        System.setProperty(Gemma3RuntimeMode.PROPERTY, "native-warp"); // legacy flag set...
        try {
            // ...but the decision is purely backend-based and ignores it.
            assertFalse(SummarizerPanel.gemmaUsesNativeWarp(Backend.AUTO),
                    "legacy -Dgemma.runtime must not force the native path when Backend != WARP");
            assertTrue(SummarizerPanel.gemmaUsesNativeWarp(Backend.WARP));
        } finally {
            if (prev == null) {
                System.clearProperty(Gemma3RuntimeMode.PROPERTY);
            } else {
                System.setProperty(Gemma3RuntimeMode.PROPERTY, prev);
            }
        }
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
