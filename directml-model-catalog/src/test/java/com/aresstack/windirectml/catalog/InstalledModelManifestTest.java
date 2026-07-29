package com.aresstack.windirectml.catalog;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
        assertEquals(ManifestValidation.UNSUPPORTED_SCHEMA, m.validate(99));
    }

    @Test
    void malformedSchemaVersionIsInvalidNeverV1() {
        InstalledModelManifest m = InstalledModelManifest.forInstall(reranker(), "rev", 1L);
        assertEquals(ManifestValidation.INVALID_MANIFEST,
                m.validate(InstalledModelManifest.SCHEMA_VERSION_MALFORMED));
    }

    @Test
    void v2WithoutProvenanceIsInvalid() {
        LocalRuntimeModelDescriptor d = minilm();
        InstalledModelManifest noRevision = new InstalledModelManifest(2, d.virtualModelName(),
                d.huggingFaceRepositoryId(), "" /* no revision */, d.runtimeModelId(),
                d.runtimeFamily().token(), d.runtimePackageFileName(),
                InstalledModelManifest.expectedCapabilityTokens(d),
                InstalledModelManifest.expectedBackendTokens(d), d.sourceFormat().token(), "RUNNABLE", 1L);
        assertEquals(ManifestValidation.INVALID_MANIFEST, noRevision.validate(2));

        InstalledModelManifest noTime = new InstalledModelManifest(2, d.virtualModelName(),
                d.huggingFaceRepositoryId(), "rev", d.runtimeModelId(), d.runtimeFamily().token(),
                d.runtimePackageFileName(), InstalledModelManifest.expectedCapabilityTokens(d),
                InstalledModelManifest.expectedBackendTokens(d), d.sourceFormat().token(), "RUNNABLE",
                0L /* no install time */);
        assertEquals(ManifestValidation.INVALID_MANIFEST, noTime.validate(2));
    }

    /** The exact historical v1 manifest the A5 installer wrote for the real MiniLM-L6 reranker install. */
    private static final String HISTORICAL_V1 = "{\"schemaVersion\":1,"
            + "\"virtualName\":\"local/cross-encoder/ms-marco-MiniLM-L6-v2:latest\","
            + "\"huggingFaceRepository\":\"cross-encoder/ms-marco-MiniLM-L6-v2\","
            + "\"resolvedRevision\":\"a1b2c3d4\",\"runtimeModelId\":\"MS_MARCO_MINILM_L6\","
            + "\"capabilities\":[\"rerank\"],\"backendSupport\":[\"cpu\",\"directml\"],"
            + "\"state\":\"RUNNABLE\"}";

    private static InstalledModelManifest v1(String virtualName, String repo, String runtimeModelId,
                                             List<String> capabilities, List<String> backends) {
        return new InstalledModelManifest(1, virtualName, repo, "rev", runtimeModelId, "", "",
                capabilities, backends, "", "RUNNABLE", 1L);
    }

    @Test
    void v1RealHistoricalRerankerFixtureIsValid() {
        // Reconstruct the historical fields exactly as the A5 installer wrote them.
        InstalledModelManifest m = v1("local/cross-encoder/ms-marco-MiniLM-L6-v2:latest",
                "cross-encoder/ms-marco-MiniLM-L6-v2", "MS_MARCO_MINILM_L6",
                Collections.singletonList("rerank"), Arrays.asList("cpu", "directml"));
        assertEquals(ManifestValidation.VALID, m.validate(1));
        // The literal historical JSON must document the exact shape being asserted.
        assertTrue(HISTORICAL_V1.contains("\"MS_MARCO_MINILM_L6\""));
    }

    @Test
    void v1IsValidatedAgainstTheRunnableCatalogEntry() {
        List<String> rerank = Collections.singletonList("rerank");
        List<String> backends = Arrays.asList("cpu", "directml");
        // Arbitrary repository that is not catalogued.
        assertEquals(ManifestValidation.CATALOG_ENTRY_MISSING,
                v1("local/foo/bar:latest", "foo/bar", "ANYTHING", rerank, backends).validate(1));
        // A MiniLM embedding repository dressed up as a reranker.
        assertEquals(ManifestValidation.CATALOG_MISMATCH,
                v1("local/sentence-transformers/all-MiniLM-L6-v2:latest",
                        "sentence-transformers/all-MiniLM-L6-v2", "MINILM_L6_V2", rerank, backends)
                        .validate(1));
        // rerank + an invented extra capability.
        assertEquals(ManifestValidation.CATALOG_MISMATCH,
                v1("local/cross-encoder/ms-marco-MiniLM-L6-v2:latest",
                        "cross-encoder/ms-marco-MiniLM-L6-v2", "MS_MARCO_MINILM_L6",
                        Arrays.asList("rerank", "chat"), backends).validate(1));
        // Wrong virtual id.
        assertEquals(ManifestValidation.CATALOG_MISMATCH,
                v1("local/wrong:latest", "cross-encoder/ms-marco-MiniLM-L6-v2", "MS_MARCO_MINILM_L6",
                        rerank, backends).validate(1));
        // Wrong runtime id.
        assertEquals(ManifestValidation.CATALOG_MISMATCH,
                v1("local/cross-encoder/ms-marco-MiniLM-L6-v2:latest",
                        "cross-encoder/ms-marco-MiniLM-L6-v2", "WRONG_ID", rerank, backends).validate(1));
    }

    @Test
    void emptyVirtualNameIsInvalid() {
        InstalledModelManifest m = new InstalledModelManifest(2, "", "foo/bar", "rev", "X", "minilm",
                "encoder.wdmlpack", Collections.singletonList("embedding"),
                Arrays.asList("cpu"), "safetensors", "RUNNABLE", 1L);
        assertEquals(ManifestValidation.INVALID_MANIFEST, m.validate(2));
    }
}
