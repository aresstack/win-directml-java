package com.aresstack.windirectml.catalog;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The hardened manifest reader: a v2 manifest is trusted only when state=RUNNABLE and every runtime fact
 * matches the catalog; a missing state is never implicitly RUNNABLE for v2; unknown schemas are not loaded;
 * v1 accepts only the historical reranker format.
 */
class InstalledModelManifestTest {

    private static LocalRuntimeModelDescriptor minilm() {
        return LocalModelCatalog.findByRepositoryId("sentence-transformers/all-MiniLM-L6-v2");
    }

    private static LocalRuntimeModelDescriptor reranker() {
        return LocalModelCatalog.findByRepositoryId("cross-encoder/ms-marco-MiniLM-L6-v2");
    }

    @Test
    void forInstallProducesAValidV2Manifest() {
        assertEquals(ManifestValidation.VALID,
                InstalledModelManifest.forInstall(minilm(), "rev", 1L).validate(2));
        assertEquals(ManifestValidation.VALID,
                InstalledModelManifest.forInstall(reranker(), "rev", 1L).validate(2));
    }

    @Test
    void v2WithoutStateIsInvalidNeverImplicitlyRunnable() {
        LocalRuntimeModelDescriptor d = minilm();
        InstalledModelManifest m = new InstalledModelManifest(2, d.virtualModelName(),
                d.huggingFaceRepositoryId(), "rev", d.runtimeModelId(), d.runtimeFamily().token(),
                d.runtimePackageFileName(), InstalledModelManifest.expectedCapabilityTokens(d),
                InstalledModelManifest.expectedBackendTokens(d), d.sourceFormat().token(),
                "" /* no state */, 1L);
        assertEquals(ManifestValidation.INVALID_MANIFEST, m.validate(2));
    }

    @Test
    void v2WithInventedChatCapabilityForMiniLmIsRejected() {
        LocalRuntimeModelDescriptor d = minilm();
        InstalledModelManifest m = new InstalledModelManifest(2, d.virtualModelName(),
                d.huggingFaceRepositoryId(), "rev", d.runtimeModelId(), d.runtimeFamily().token(),
                d.runtimePackageFileName(), Arrays.asList("embedding", "chat"),
                InstalledModelManifest.expectedBackendTokens(d), d.sourceFormat().token(), "RUNNABLE", 1L);
        assertEquals(ManifestValidation.CATALOG_MISMATCH, m.validate(2));
    }

    @Test
    void v2WithWrongPackageNameIsRejected() {
        LocalRuntimeModelDescriptor d = minilm();
        InstalledModelManifest m = new InstalledModelManifest(2, d.virtualModelName(),
                d.huggingFaceRepositoryId(), "rev", d.runtimeModelId(), d.runtimeFamily().token(),
                "wrong.wdmlpack", InstalledModelManifest.expectedCapabilityTokens(d),
                InstalledModelManifest.expectedBackendTokens(d), d.sourceFormat().token(), "RUNNABLE", 1L);
        assertEquals(ManifestValidation.CATALOG_MISMATCH, m.validate(2));
    }

    @Test
    void v2ForANonCatalogRepositoryIsCatalogEntryMissing() {
        InstalledModelManifest m = new InstalledModelManifest(2, "local/foo/bar:latest", "foo/bar", "rev",
                "FOO", "minilm", "encoder.wdmlpack", Collections.singletonList("embedding"),
                Arrays.asList("cpu", "directml"), "safetensors", "RUNNABLE", 1L);
        assertEquals(ManifestValidation.CATALOG_ENTRY_MISSING, m.validate(2));
    }

    @Test
    void v2PointingAtTheUnverifiedL12IsRejected() {
        LocalRuntimeModelDescriptor l12 =
                LocalModelCatalog.findByRepositoryId("cross-encoder/ms-marco-MiniLM-L12-v2");
        InstalledModelManifest m = new InstalledModelManifest(2, l12.virtualModelName(),
                l12.huggingFaceRepositoryId(), "rev", l12.runtimeModelId(), l12.runtimeFamily().token(),
                l12.runtimePackageFileName(), Collections.singletonList("rerank"),
                Arrays.asList("cpu", "directml"), "safetensors", "RUNNABLE", 1L);
        // Descriptor exists but is UNVERIFIED (not runnable) -> not trusted.
        assertEquals(ManifestValidation.CATALOG_MISMATCH, m.validate(2));
    }

    @Test
    void unknownSchemaIsNeverLoaded() {
        InstalledModelManifest m = InstalledModelManifest.forInstall(minilm(), "rev", 1L);
        assertEquals(ManifestValidation.UNSUPPORTED_SCHEMA, m.validate(3));
        assertEquals(ManifestValidation.UNSUPPORTED_SCHEMA, m.validate(0));
    }

    @Test
    void v1AcceptsOnlyTheHistoricalRerankerFormat() {
        // Historical reranker: rerank capability, missing state = historical RUNNABLE.
        InstalledModelManifest ok = new InstalledModelManifest(1,
                "local/cross-encoder/ms-marco-MiniLM-L6-v2:latest", "cross-encoder/ms-marco-MiniLM-L6-v2",
                "rev", "MS_MARCO_MINILM_L6", "", "", Collections.singletonList("rerank"),
                Arrays.asList("cpu", "directml"), "", "", 1L);
        assertEquals(ManifestValidation.VALID, ok.validate(1));

        // A v1 manifest without the rerank capability is not the known historical format.
        List<String> embedding = Collections.singletonList("embedding");
        InstalledModelManifest notReranker = new InstalledModelManifest(1, "local/x:latest", "x/y", "rev",
                "X", "", "", embedding, Collections.<String>emptyList(), "", "", 1L);
        assertEquals(ManifestValidation.INVALID_MANIFEST, notReranker.validate(1));
    }

    @Test
    void emptyVirtualNameIsInvalid() {
        InstalledModelManifest m = new InstalledModelManifest(2, "", "foo/bar", "rev", "X", "minilm",
                "encoder.wdmlpack", Collections.singletonList("embedding"),
                Arrays.asList("cpu"), "safetensors", "RUNNABLE", 1L);
        assertEquals(ManifestValidation.INVALID_MANIFEST, m.validate(2));
    }
}
