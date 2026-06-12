package com.aresstack.windirectml.inference.model;

import com.aresstack.windirectml.inference.smollm2.SmolLM2RuntimeLoadability;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Shared neutral runtime-loadability report. Device-free. Also shows a decoder-only family report maps onto it 1:1
 * (same field shape), proving the type is usable by both without changing the decoder-only block.
 */
class RuntimeLoadabilityTest {

    @Test
    void factoriesSetFields() {
        RuntimeLoadability loadable = RuntimeLoadability.loadable("reference", "ok");
        assertTrue(loadable.runtimeLoadable());
        assertEquals("reference", loadable.runtimeLoadMode());
        assertEquals("ok", loadable.runtimeLoadableReason());

        RuntimeLoadability notLoadable = RuntimeLoadability.notLoadable("manifest-only", "no payload");
        assertFalse(notLoadable.runtimeLoadable());
        assertEquals("manifest-only", notLoadable.runtimeLoadMode());
    }

    @Test
    void rejectsBlankModeOrReason() {
        assertThrows(IllegalArgumentException.class, () -> new RuntimeLoadability(true, "", "ok"));
        assertThrows(IllegalArgumentException.class, () -> new RuntimeLoadability(true, "mode", " "));
    }

    @Test
    void decoderOnlyLoadabilityMapsOntoTheNeutralReport() {
        SmolLM2RuntimeLoadability smol = SmolLM2RuntimeLoadability.loadable("reference", "ref ok");
        RuntimeLoadability mapped = new RuntimeLoadability(
                smol.runtimeLoadable(), smol.runtimeLoadMode(), smol.runtimeLoadableReason());

        assertTrue(mapped.runtimeLoadable());
        assertEquals("reference", mapped.runtimeLoadMode());
        assertEquals("ref ok", mapped.runtimeLoadableReason());
    }
}
