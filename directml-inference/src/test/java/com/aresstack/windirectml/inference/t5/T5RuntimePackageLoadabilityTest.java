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

        // Manifest-only package: no payload -> not runtime-loadable, not executable, honest mode/reason (T5-1).
        assertFalse(loadability.runtimeLoadable());
        assertFalse(runtimePackage.executable());
        assertEquals(T5ManifestPayloadPolicy.MODE_MANIFEST_ONLY, loadability.runtimeLoadMode());
        assertEquals(T5ManifestPayloadPolicy.REASON_MANIFEST_ONLY, loadability.runtimeLoadableReason());
        assertEquals(runtimePackage.runtimeLoadable(), loadability.runtimeLoadable());
    }
}
