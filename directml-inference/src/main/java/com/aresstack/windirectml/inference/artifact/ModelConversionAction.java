package com.aresstack.windirectml.inference.artifact;

/**
 * The state-dependent action offered by the unified "Convert / Status" button.
 */
public enum ModelConversionAction {
    /** No package exists yet; build one from the raw source. */
    CONVERT("Convert"),
    /** A package exists but is stale or the operator wants a fresh build. */
    RECONVERT("Reconvert"),
    /** A package exists but is corrupt / not loadable / not executable; rebuild it. */
    REPAIR("Repair package"),
    /** Nothing to convert (e.g. raw source missing); only show the current status. */
    INSPECT("Inspect"),
    /** No package compiler exists for this family yet. */
    NOT_SUPPORTED("Package compiler not implemented for this family");

    private final String label;

    ModelConversionAction(String label) {
        this.label = label;
    }

    /** Human label suitable for the conversion button. */
    public String label() {
        return label;
    }

    /** Whether pressing the button should actually write a package ({@link #CONVERT}/{@link #RECONVERT}/{@link #REPAIR}). */
    public boolean writesPackage() {
        return this == CONVERT || this == RECONVERT || this == REPAIR;
    }
}
