package com.aresstack.windirectml.catalog;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * The neutral value object of an installed model's provenance manifest ({@code askai-local-model.json}),
 * shared by the Java-8 host and the Java-21 sidecar so both apply the SAME trust rules. JSON I/O stays in
 * each module (this class is dependency-free); {@link #validate(int)} enforces the schema rules and the
 * agreement with {@link LocalModelCatalog}.
 *
 * <p>A manifest is installation provenance, NOT an authority that can invent capabilities: a v2 manifest is
 * only {@link ManifestValidation#VALID} when every declared runtime fact matches the catalog descriptor.
 * A missing/empty {@code state} is historical RUNNABLE semantics for schema v1 ONLY — for v2 it is
 * {@link ManifestValidation#INVALID_MANIFEST}.</p>
 */
public final class InstalledModelManifest {

    /** The current manifest schema version. */
    public static final int CURRENT_SCHEMA_VERSION = 2;
    /** The historical reranker-only schema version. */
    public static final int LEGACY_RERANKER_SCHEMA_VERSION = 1;
    /**
     * A sentinel a codec passes for a {@code schemaVersion} field that is PRESENT but not an integer
     * (a string, a fractional number, …). It is distinct from an ABSENT field (which is historical v1):
     * a present-but-invalid version is {@link ManifestValidation#INVALID_MANIFEST}, never a silent fallback.
     */
    public static final int SCHEMA_VERSION_MALFORMED = Integer.MIN_VALUE;

    private final int schemaVersion;
    private final String virtualName;
    private final String huggingFaceRepository;
    private final String resolvedRevision;
    private final String runtimeModelId;
    private final String runtimeFamily;
    private final String runtimePackage;
    private final List<String> capabilities;
    private final List<String> supportedBackends;
    private final String sourceFormat;
    private final String state;
    private final long installedAt;

    public InstalledModelManifest(int schemaVersion, String virtualName, String huggingFaceRepository,
                                  String resolvedRevision, String runtimeModelId, String runtimeFamily,
                                  String runtimePackage, List<String> capabilities,
                                  List<String> supportedBackends, String sourceFormat, String state,
                                  long installedAt) {
        this.schemaVersion = schemaVersion;
        this.virtualName = nz(virtualName);
        this.huggingFaceRepository = nz(huggingFaceRepository);
        this.resolvedRevision = nz(resolvedRevision);
        this.runtimeModelId = nz(runtimeModelId);
        this.runtimeFamily = nz(runtimeFamily);
        this.runtimePackage = nz(runtimePackage);
        this.capabilities = unmodifiable(capabilities);
        this.supportedBackends = unmodifiable(supportedBackends);
        this.sourceFormat = nz(sourceFormat);
        this.state = nz(state); // NOT defaulted to RUNNABLE — validate() decides per schema version
        this.installedAt = installedAt;
    }

    /** Build a current-schema manifest for a verified install from the neutral catalog descriptor. */
    public static InstalledModelManifest forInstall(LocalRuntimeModelDescriptor descriptor,
                                                    String resolvedRevision, long installedAtEpochMillis) {
        return new InstalledModelManifest(CURRENT_SCHEMA_VERSION, descriptor.virtualModelName(),
                descriptor.huggingFaceRepositoryId(), resolvedRevision, descriptor.runtimeModelId(),
                descriptor.runtimeFamily().token(), descriptor.runtimePackageFileName(),
                expectedCapabilityTokens(descriptor), expectedBackendTokens(descriptor),
                descriptor.sourceFormat().token(), "RUNNABLE", installedAtEpochMillis);
    }

    /** The catalog-derived capability tokens for a descriptor, in {@link ModelCapability} order. */
    public static List<String> expectedCapabilityTokens(LocalRuntimeModelDescriptor descriptor) {
        List<String> tokens = new ArrayList<String>();
        for (ModelCapability capability : ModelCapability.values()) {
            if (descriptor.capabilities().contains(capability)) {
                tokens.add(capability.token());
            }
        }
        return tokens;
    }

    /** The catalog-derived backend tokens for a descriptor, in {@link CatalogBackend} order. */
    public static List<String> expectedBackendTokens(LocalRuntimeModelDescriptor descriptor) {
        List<String> tokens = new ArrayList<String>();
        for (CatalogBackend backend : CatalogBackend.values()) {
            if (descriptor.supportedBackends().contains(backend)) {
                tokens.add(backend.token());
            }
        }
        return tokens;
    }

    /**
     * Validate this manifest against its schema rules and the catalog. {@code declaredSchemaVersion} is the
     * schema version read from the JSON (so an absent field can be distinguished from a present one); pass
     * {@link #getSchemaVersion()} when it was parsed into this object.
     */
    public ManifestValidation validate(int declaredSchemaVersion) {
        if (declaredSchemaVersion == SCHEMA_VERSION_MALFORMED) {
            // schemaVersion present but not an integer — never a silent fallback to v1.
            return ManifestValidation.INVALID_MANIFEST;
        }
        if (virtualName.isEmpty() || huggingFaceRepository.isEmpty()) {
            return ManifestValidation.INVALID_MANIFEST;
        }
        if (declaredSchemaVersion == LEGACY_RERANKER_SCHEMA_VERSION) {
            // v1: the EXACT historical reranker format, validated against the real RUNNABLE catalog entry.
            // A missing state is historical RUNNABLE; a present state, if any, must be RUNNABLE.
            if (!state.isEmpty() && !"RUNNABLE".equalsIgnoreCase(state)) {
                return ManifestValidation.INVALID_MANIFEST;
            }
            LocalRuntimeModelDescriptor descriptor =
                    LocalModelCatalog.findByRepositoryId(huggingFaceRepository);
            if (descriptor == null) {
                return ManifestValidation.CATALOG_ENTRY_MISSING;
            }
            // Only a runnable reranker entry, with EXACTLY {rerank}, the exact virtual id + runtime id, and
            // (when the old manifest carried them) the exact expected backends.
            if (!descriptor.isRunnable()
                    || !descriptor.hasCapability(ModelCapability.RERANK)
                    || !descriptor.virtualModelName().equals(virtualName)
                    || !descriptor.runtimeModelId().equals(runtimeModelId)
                    || !sameSet(capabilities, expectedCapabilityTokens(descriptor))
                    || (!supportedBackends.isEmpty()
                            && !sameSet(supportedBackends, expectedBackendTokens(descriptor)))) {
                return ManifestValidation.CATALOG_MISMATCH;
            }
            return ManifestValidation.VALID;
        }
        if (declaredSchemaVersion == CURRENT_SCHEMA_VERSION) {
            // v2: state must be explicitly RUNNABLE; all runtime facts AND provenance must be present; then
            // they must match the catalog. No implicit RUNNABLE, no invented capabilities, no blank
            // provenance (v2 is written only by the new installer, so blank values are never legitimate).
            if (!"RUNNABLE".equalsIgnoreCase(state)
                    || runtimeFamily.isEmpty() || runtimePackage.isEmpty() || sourceFormat.isEmpty()
                    || capabilities.isEmpty() || runtimeModelId.isEmpty()
                    || resolvedRevision.isEmpty() || installedAt <= 0L) {
                return ManifestValidation.INVALID_MANIFEST;
            }
            LocalRuntimeModelDescriptor descriptor =
                    LocalModelCatalog.findByRepositoryId(huggingFaceRepository);
            if (descriptor == null) {
                return ManifestValidation.CATALOG_ENTRY_MISSING;
            }
            if (!descriptor.isRunnable()
                    || !descriptor.virtualModelName().equals(virtualName)
                    || !descriptor.runtimeModelId().equals(runtimeModelId)
                    || !descriptor.runtimeFamily().token().equals(runtimeFamily)
                    || !descriptor.runtimePackageFileName().equals(runtimePackage)
                    || !descriptor.sourceFormat().token().equals(sourceFormat)
                    || !sameSet(capabilities, expectedCapabilityTokens(descriptor))
                    || !sameSet(supportedBackends, expectedBackendTokens(descriptor))) {
                return ManifestValidation.CATALOG_MISMATCH;
            }
            return ManifestValidation.VALID;
        }
        return ManifestValidation.UNSUPPORTED_SCHEMA;
    }

    public int getSchemaVersion() { return schemaVersion; }
    public String getVirtualName() { return virtualName; }
    public String getHuggingFaceRepository() { return huggingFaceRepository; }
    public String getResolvedRevision() { return resolvedRevision; }
    public String getRuntimeModelId() { return runtimeModelId; }
    public String getRuntimeFamily() { return runtimeFamily; }
    public String getRuntimePackage() { return runtimePackage; }
    public List<String> getCapabilities() { return capabilities; }
    public List<String> getSupportedBackends() { return supportedBackends; }
    public String getSourceFormat() { return sourceFormat; }
    public String getState() { return state; }
    public long getInstalledAt() { return installedAt; }

    public boolean hasCapability(ModelCapability capability) {
        return capabilities.contains(capability.token());
    }

    // ------------------------------------------------------------------ helpers

    private static boolean sameSet(List<String> a, List<String> b) {
        Set<String> sa = new HashSet<String>();
        for (String s : a) {
            sa.add(s.toLowerCase(Locale.ROOT));
        }
        Set<String> sb = new HashSet<String>();
        for (String s : b) {
            sb.add(s.toLowerCase(Locale.ROOT));
        }
        return sa.equals(sb);
    }

    private static String nz(String value) {
        return value == null ? "" : value.trim();
    }

    private static List<String> unmodifiable(List<String> in) {
        return in == null ? Collections.<String>emptyList()
                : Collections.unmodifiableList(new ArrayList<String>(in));
    }
}
