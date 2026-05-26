package com.aresstack.windirectml.runtime.api;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Public configuration for loading a reranker model.
 *
 * <p>The {@code modelDir} is optional: when omitted, it is derived from
 * {@link RerankerModelId#directoryName()} as a relative path (e.g.
 * {@code Path.of("model/cross-encoder-ms-marco-MiniLM-L-6-v2")}). When
 * provided explicitly it is validated against the expected directory name so
 * that a mismatch between the strong identifier and the path is caught at
 * configuration time rather than at model-loading time.
 */
public final class RerankerConfig {
    private final RerankerModelId model;
    private final Path modelDir;

    private RerankerConfig(Builder builder) {
        this.model = Objects.requireNonNull(builder.model, "model");
        if (builder.modelDir == null) {
            this.modelDir = Path.of("model").resolve(model.directoryName());
        } else {
            validateModelDir(builder.modelDir, this.model);
            this.modelDir = builder.modelDir;
        }
    }

    /** Validates that the explicit modelDir ends with the expected directory name for the model. */
    private static void validateModelDir(Path modelDir, RerankerModelId model) {
        String lastName = modelDir.getFileName() == null ? "" : modelDir.getFileName().toString();
        if (!lastName.equals(model.directoryName())) {
            throw new IllegalArgumentException(
                    "modelDir '" + modelDir + "' does not end with the expected directory name '"
                            + model.directoryName() + "' for model " + model.name()
                            + ". Either omit modelDir to use the default, or ensure the final"
                            + " path component matches the model's canonical directory name.");
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public RerankerModelId model() {
        return model;
    }

    public Path modelDir() {
        return modelDir;
    }

    com.aresstack.windirectml.runtime.facade.RerankerModelConfig toFacade() {
        return new com.aresstack.windirectml.runtime.facade.RerankerModelConfig(modelDir);
    }

    public static final class Builder {
        private RerankerModelId model;
        private Path modelDir;

        private Builder() { }

        public Builder model(RerankerModelId model) {
            this.model = Objects.requireNonNull(model, "model");
            return this;
        }

        /**
         * Override the model directory path.
         *
         * <p>When set, the final path component must match
         * {@link RerankerModelId#directoryName()} for the chosen model. If
         * omitted, the directory is derived automatically as
         * {@code model/<directoryName>}.
         */
        public Builder modelDir(Path modelDir) {
            this.modelDir = Objects.requireNonNull(modelDir, "modelDir");
            return this;
        }

        public RerankerConfig build() {
            return new RerankerConfig(this);
        }
    }
}
