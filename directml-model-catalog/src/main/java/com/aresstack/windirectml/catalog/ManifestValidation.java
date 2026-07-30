package com.aresstack.windirectml.catalog;

/**
 * The typed outcome of validating an installed-model manifest against the schema rules and the neutral
 * catalog. The manifest is installation PROVENANCE, never an authority that can invent capabilities: a v2
 * manifest is only trusted when every declared runtime fact matches the catalog descriptor.
 *
 * <p>{@link #PACKAGE_MISSING} and {@link #PACKAGE_NOT_LOADABLE} are produced by the runtime load step (the
 * sidecar), not by the pure manifest reader; they are part of the same typed vocabulary so callers can
 * report one consistent set of reasons.</p>
 */
public enum ManifestValidation {

    /** Schema + catalog agreement all hold; safe to publish/run. */
    VALID,
    /** Structurally unreadable, missing a required field, or a v2 manifest without state=RUNNABLE. */
    INVALID_MANIFEST,
    /** schemaVersion is neither the historical 1 nor the current 2. Never loaded, never in /api/tags. */
    UNSUPPORTED_SCHEMA,
    /** The manifest names a repository that is not in the catalog. */
    CATALOG_ENTRY_MISSING,
    /** A declared runtime fact (family/package/capabilities/backends/sourceFormat/id) contradicts the catalog. */
    CATALOG_MISMATCH,
    /** The compiled runtime package file is absent (runtime load step). */
    PACKAGE_MISSING,
    /** The compiled runtime package exists but cannot be opened/loaded (runtime load step). */
    PACKAGE_NOT_LOADABLE;

    public boolean isValid() {
        return this == VALID;
    }
}
