package com.aresstack.windirectml.inference.gemma;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GEMMA-WARP-13b-4: execution-mode selection. The native-warp path defaults to the resident/batched path;
 * {@code -Dgemma.warp.execution=sync} selects the float[] debug/fallback path.
 */
class Gemma3WarpExecutionModeTest {

    @Test
    void defaultsToResidentWhenUnset() {
        String prev = System.getProperty(Gemma3WarpExecutionMode.PROPERTY);
        System.clearProperty(Gemma3WarpExecutionMode.PROPERTY);
        try {
            assertEquals(Gemma3WarpExecutionMode.RESIDENT, Gemma3WarpExecutionMode.fromSystemProperty());
        } finally {
            if (prev != null) {
                System.setProperty(Gemma3WarpExecutionMode.PROPERTY, prev);
            }
        }
    }

    @Test
    void syncFlagSelectsSyncFallback() {
        String prev = System.getProperty(Gemma3WarpExecutionMode.PROPERTY);
        System.setProperty(Gemma3WarpExecutionMode.PROPERTY, "sync");
        try {
            assertEquals(Gemma3WarpExecutionMode.SYNC, Gemma3WarpExecutionMode.fromSystemProperty());
        } finally {
            if (prev == null) {
                System.clearProperty(Gemma3WarpExecutionMode.PROPERTY);
            } else {
                System.setProperty(Gemma3WarpExecutionMode.PROPERTY, prev);
            }
        }
    }

    @Test
    void valueParsing() {
        assertEquals(Gemma3WarpExecutionMode.RESIDENT, Gemma3WarpExecutionMode.fromValue(null));
        assertEquals(Gemma3WarpExecutionMode.RESIDENT, Gemma3WarpExecutionMode.fromValue(""));
        assertEquals(Gemma3WarpExecutionMode.RESIDENT, Gemma3WarpExecutionMode.fromValue("resident"));
        assertEquals(Gemma3WarpExecutionMode.RESIDENT, Gemma3WarpExecutionMode.fromValue("batched"));
        assertEquals(Gemma3WarpExecutionMode.RESIDENT, Gemma3WarpExecutionMode.fromValue("something-else"));
        assertEquals(Gemma3WarpExecutionMode.SYNC, Gemma3WarpExecutionMode.fromValue("sync"));
        assertEquals(Gemma3WarpExecutionMode.SYNC, Gemma3WarpExecutionMode.fromValue("SYNC"));
        assertEquals(Gemma3WarpExecutionMode.SYNC, Gemma3WarpExecutionMode.fromValue("legacy"));
        assertEquals(Gemma3WarpExecutionMode.SYNC, Gemma3WarpExecutionMode.fromValue("float"));
        assertTrue(Gemma3WarpExecutionMode.RESIDENT.isResident());
        assertFalse(Gemma3WarpExecutionMode.SYNC.isResident());
    }
}
