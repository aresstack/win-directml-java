package com.aresstack.windirectml.sidecar.workbench.panels;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Verifies that the {@link HealthPanel} formats the fallback signal
 * coming from the {@code health} response in a way that surfaces both
 * the boolean flag and the human-readable reason.
 */
class HealthPanelFallbackFormatTest {

    @Test
    void formatsFallbackFalseAsFalse() {
        assertEquals("false", HealthPanel.formatFallback(false, null));
        assertEquals("false", HealthPanel.formatFallback(false, "ignored"));
    }

    @Test
    void formatsFallbackTrueWithReason() {
        assertEquals("true – DirectML.dll missing",
                HealthPanel.formatFallback(true, "DirectML.dll missing"));
    }

    @Test
    void formatsFallbackTrueWithoutReasonAsPlainTrue() {
        assertEquals("true", HealthPanel.formatFallback(true, null));
        assertEquals("true", HealthPanel.formatFallback(true, ""));
    }
}
