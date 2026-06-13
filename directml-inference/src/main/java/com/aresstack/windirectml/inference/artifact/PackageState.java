package com.aresstack.windirectml.inference.artifact;

/**
 * State of the compiled runtime package ({@code .wdmlpack}) for a model.
 */
public enum PackageState {
    /** No runtime package file exists. */
    PACKAGE_MISSING,
    /** A package exists but is older than the current compiler/schema or the source weights changed. */
    PACKAGE_STALE,
    /** A package file exists but cannot be opened/parsed. */
    PACKAGE_CORRUPT,
    /** The package opens but its weights/runtime structures are not loadable. */
    PACKAGE_NOT_LOADABLE,
    /** The package is loadable but generation is not certified/executable. */
    PACKAGE_NOT_EXECUTABLE,
    /** The package is present, loadable and executable. */
    PACKAGE_VALID,
    /**
     * The family has no {@code .wdmlpack} compiler yet, so it cannot be converted or executed. This is
     * a homogeneous <b>not-executable</b> state (encoder/reranker/Phi-3 today) - never a legacy runtime
     * path that silently loads raw weights.
     */
    PACKAGE_COMPILER_MISSING
}
