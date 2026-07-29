package com.aresstack.windirectml.catalog;

/**
 * One file an install strategy must fetch from the source repository, expressed neutrally (no URL, no I/O).
 * {@link #remotePath()} is relative to the repository root (it may include a subdirectory, e.g. the Phi-3
 * ONNX INT4 folder); {@link #localName()} is the name it is stored under in the model directory.
 *
 * <p>Java-8 compatible.</p>
 */
public final class DownloadFile {

    private final String remotePath;
    private final String localName;
    private final boolean required;

    public DownloadFile(String remotePath, String localName, boolean required) {
        if (remotePath == null || remotePath.trim().isEmpty()) {
            throw new IllegalArgumentException("remotePath must not be blank");
        }
        this.remotePath = remotePath.trim();
        this.localName = localName == null || localName.trim().isEmpty()
                ? this.remotePath : localName.trim();
        this.required = required;
    }

    /** A required file at the repository root, stored under its own name. */
    public static DownloadFile required(String name) {
        return new DownloadFile(name, name, true);
    }

    /** An optional file at the repository root, stored under its own name. */
    public static DownloadFile optional(String name) {
        return new DownloadFile(name, name, false);
    }

    public String remotePath() {
        return remotePath;
    }

    public String localName() {
        return localName;
    }

    public boolean required() {
        return required;
    }

    @Override
    public String toString() {
        return localName + (required ? " (required)" : " (optional)");
    }
}
