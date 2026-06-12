package com.aresstack.windirectml.inference.simd;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyReferenceDenseOps;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * No-Vector-module loadability gate. Runs ONLY when the test JVM is started without
 * {@code --add-modules=jdk.incubator.vector} and {@code -Dwindirectml.test.noVectorModule=true} is set (the property is
 * forwarded to the test JVM). In normal CI (module present, property unset) it is skipped.
 *
 * <p>Proves that the always-loaded facades no longer have a hard Vector-API load dependency: they load without a
 * {@code NoClassDefFoundError} and fall back to the scalar provider with correct results.</p>
 */
class NoVectorModuleLoadabilityTest {

    @Test
    void facadesLoadAndFallBackToScalarWithoutVectorModule() throws Exception {
        assumeTrue(Boolean.getBoolean("windirectml.test.noVectorModule"),
                "Only runs in a JVM started without --add-modules=jdk.incubator.vector");

        // Loading these must NOT throw NoClassDefFoundError: jdk/incubator/vector/Vector anymore.
        assertNotNull(Class.forName("com.aresstack.windirectml.inference.qwen.SimdOps"));
        assertNotNull(Class.forName("com.aresstack.windirectml.inference.phi3.SimdOps"));
        assertNotNull(Class.forName("com.aresstack.windirectml.inference.decoderonly.DecoderOnlyReferenceDenseOps"));
        assertNotNull(Class.forName("com.aresstack.windirectml.inference.simd.SimdMath"));

        // Without the module the provider is the scalar fallback.
        assertFalse(SimdMath.provider().enabled());
        assertFalse(DecoderOnlyReferenceDenseOps.enabled());

        // Scalar math is still correct.
        assertEquals(32.0f, DecoderOnlyReferenceDenseOps.dot(
                new float[]{1, 2, 3}, 0, new float[]{4, 5, 6}, 0, 3), 1e-4f);
    }
}
