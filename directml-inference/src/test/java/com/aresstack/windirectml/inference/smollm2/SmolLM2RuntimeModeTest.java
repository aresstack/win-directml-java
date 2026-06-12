package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyRuntimeMode;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The SmolLM2 runtime mode keeps its family-specific display labels but maps cleanly onto the shared
 * {@link DecoderOnlyRuntimeMode} adapter point.
 */
class SmolLM2RuntimeModeTest {

    @Test
    void mapsEachModeOntoTheGenericRuntimeMode() {
        assertEquals(DecoderOnlyRuntimeMode.REFERENCE, SmolLM2RuntimeMode.REFERENCE.toDecoderOnlyRuntimeMode());
        assertEquals(DecoderOnlyRuntimeMode.WARP, SmolLM2RuntimeMode.WARP.toDecoderOnlyRuntimeMode());
        assertEquals(DecoderOnlyRuntimeMode.AUTO, SmolLM2RuntimeMode.AUTO.toDecoderOnlyRuntimeMode());
    }

    @Test
    void keepsItsOwnSpecificDisplayLabels() {
        // SmolLM2 must stay honest about its CPU/GPU split, not inherit the generic label.
        assertEquals("warp (WARP projection path; norms/RoPE/attention/KV-cache on CPU)",
                SmolLM2RuntimeMode.WARP.displayLabel());
    }
}
