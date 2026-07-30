package com.aresstack.windirectml.inference.api;

/**
 * Controls whether a model may be loaded only from a compiled runtime package, or whether the
 * runtime may compile/import from raw weights when the package is absent.
 *
 * <p>The production contract for AskAI is {@link #PACKAGE_ONLY}: a family runtime must load from its
 * {@code model_*.wdmlpack} and must never silently fall back to raw safetensors/ONNX. Under this
 * policy a missing package fails with {@link GenerationErrorCode#PACKAGE_MISSING} and a loader that
 * would internally reconstruct from raw weights fails with
 * {@link GenerationErrorCode#RAW_WEIGHTS_FALLBACK_BLOCKED} rather than quietly loading them.
 */
public enum LoadPolicy {

    /**
     * Load exclusively from the compiled {@code *.wdmlpack}. No compilation, no raw-weight fallback.
     * This is the production default for the AskAI sidecar.
     */
    PACKAGE_ONLY,

    /**
     * Permit compiling/importing the runtime package from raw weights when it does not yet exist.
     * Intended for developer/workbench first-run flows, never for the fail-closed production path.
     */
    ALLOW_COMPILE
}
