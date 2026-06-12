package com.aresstack.windirectml.inference.t5;

/**
 * Honest T5 package status vocabulary (todo item T5-1).
 *
 * <p>Replaces the previous blanket {@code runtimeLoadable=false} / "T5 runtime is not implemented yet" with a clear
 * separation: a package can be <b>weightsLoadable</b> (payload + complete layout + supported dtypes),
 * <b>runtimeLoadable</b> (the loader opens it and builds the runtime structures — for T5 this holds whenever the
 * weights load), and <b>executable</b> (the engine can run a certified generation). T5 generation is not yet certified,
 * so {@code executable} stays {@code false}; the runtime-load status is no longer hidden behind a false
 * {@code runtimeLoadable}.</p>
 */
final class T5ManifestPayloadPolicy {

    private T5ManifestPayloadPolicy() {
    }

    /** Default for metadata-only packages (no weights payload). */
    static final boolean PAYLOAD_INCLUDED = false;

    // ── runtimeLoadMode values ──────────────────────────────────────────────
    /** No weights payload in the package. */
    static final String MODE_MANIFEST_ONLY = "t5-manifest-only";
    /** Payload present but weights are not loadable (incomplete layout or unsupported runtime dtype). */
    static final String MODE_WEIGHTS_NOT_LOADABLE = "t5-weights-not-loadable";
    /** Weights + runtime structures load, but generation is not yet certified. */
    static final String MODE_RUNTIME_LOADABLE_NOT_EXECUTABLE = "t5-runtime-loadable-not-executable";
    /** Reserved for when T5 generation is certified (set by a later T5-engine slice). */
    static final String MODE_EXECUTABLE = "t5-executable";

    // ── reason strings ──────────────────────────────────────────────────────
    static final String REASON_MANIFEST_ONLY =
            "manifest-only package: weights payload not included";
    static final String REASON_WEIGHTS_NOT_LOADABLE =
            "weights not loadable: incomplete T5 layout or unsupported runtime dtype";
    static final String REASON_RUNTIME_LOADABLE_NOT_EXECUTABLE =
            "runtime loads weights and structures, but T5 generation is not yet certified";
    static final String REASON_EXECUTABLE = "T5 generation certified";
}
