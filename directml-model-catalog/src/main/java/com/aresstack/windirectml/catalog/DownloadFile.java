package com.aresstack.windirectml.catalog;

/**
 * One file an install strategy must fetch from the source repository, expressed neutrally (no URL, no I/O).
 *
 * <p>{@link #remotePath()} is ALWAYS complete and relative to the repository ROOT — it already includes any
 * source subdirectory (e.g. {@code onnx/model_q4f16.onnx}, {@code directml/directml-int4-awq-block-128/model.onnx}).
 * A downloader must resolve exactly {@code <repositoryId>/<remotePath>} and must NEVER prepend any further
 * subdirectory, otherwise it would produce a doubled prefix like {@code onnx/onnx/model_q4f16.onnx}.</p>
 *
 * <p>{@link #localName()} is the flat name the file is stored under in the local model directory (the source
 * subdirectory is intentionally dropped so the runtime sees one flat directory).</p>
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
