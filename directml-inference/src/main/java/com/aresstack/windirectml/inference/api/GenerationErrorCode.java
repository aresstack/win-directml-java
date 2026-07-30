package com.aresstack.windirectml.inference.api;

/**
 * Typed failure categories for the generation runtime. AskAI maps these onto its own transport
 * error surface (e.g. HTTP status / JSON error codes) without inspecting exception messages.
 */
public enum GenerationErrorCode {

    /** The request or its arguments were malformed (null descriptor, blank prompt, etc.). */
    INVALID_REQUEST,

    /**
     * The requested backend is not in the catalog's allowed-backend matrix for the model's family.
     * Example: Gemma&nbsp;3 with {@code CPU} (Gemma runs only via the native WARP/DirectML path).
     */
    UNSUPPORTED_BACKEND,

    /** The model directory does not exist or is not a directory. */
    MODEL_DIRECTORY_NOT_FOUND,

    /** No adapter is registered for the descriptor's runtime family. */
    UNSUPPORTED_FAMILY,

    /** Package-only loading was requested but the compiled {@code *.wdmlpack} is absent. */
    PACKAGE_MISSING,

    /** The runtime package exists but could not be opened/parsed as a loadable package. */
    PACKAGE_NOT_LOADABLE,

    /**
     * A package-backed load would have fallen back to raw safetensors/ONNX weights, which the
     * package-only policy forbids.
     */
    RAW_WEIGHTS_FALLBACK_BLOCKED,

    /** A required tokenizer / configuration asset was missing from the model directory. */
    MODEL_ASSETS_MISSING,

    /**
     * Access to a gated model was blocked by an external precondition (e.g. a missing Hugging Face
     * token for Gemma). Distinct from a genuine runtime failure: the model itself may be fine.
     */
    GATED_ACCESS_BLOCKED,

    /** The runtime failed to initialize (device/package/weights) for a supported configuration. */
    INITIALIZATION_FAILED,

    /** Generation started but failed partway through. */
    GENERATION_FAILED
}
