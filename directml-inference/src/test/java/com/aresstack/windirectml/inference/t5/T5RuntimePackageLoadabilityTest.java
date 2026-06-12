package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.model.RuntimeLoadability;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * T5 exposes its package loadability through the shared neutral {@link RuntimeLoadability} report. Device-free:
 * uses a metadata-only T5 package (no payload, no device).
 */
class T5RuntimePackageLoadabilityTest {

    @Test
    void metadataOnlyPackageReportsLoadabilityViaSharedReport() {
        T5PackageMetadata metadata = T5PackageMetadata.from(T5TestFixtures.untiedConfig());
        T5RuntimePackage runtimePackage = T5RuntimePackage.fromMetadata(metadata);

        RuntimeLoadability loadability = runtimePackage.loadability();

        // Faithful view of the (pre-existing) manifest fields for a manifest-only package.
        assertFalse(loadability.runtimeLoadable());
        assertEquals("not-implemented", loadability.runtimeLoadMode());
        assertEquals("T5 runtime is not implemented yet", loadability.runtimeLoadableReason());
        assertEquals(runtimePackage.runtimeLoadable(), loadability.runtimeLoadable());
    }
}
