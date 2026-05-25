package com.aresstack.windirectml.runtime.facade;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Configuration for creating a {@link LocalMlRuntime}.
 * <p>
 * Use the {@link #builder()} to construct an instance:
 * <pre>{@code
 * LocalMlRuntimeConfig config = LocalMlRuntimeConfig.builder()
 *         .backend(Backend.AUTO)
 *         .build();
 * }</pre>
 */
public final class LocalMlRuntimeConfig {

    private final Backend backend;

    private LocalMlRuntimeConfig(Builder builder) {
        this.backend = builder.backend;
    }

    /** The hardware backend selection. */
    public Backend backend() {
        return backend;
    }

    /** Create a new builder with defaults ({@link Backend#AUTO}). */
    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private Backend backend = Backend.AUTO;

        private Builder() {}

        /** Set the hardware backend. */
        public Builder backend(Backend backend) {
            this.backend = Objects.requireNonNull(backend, "backend");
            return this;
        }

        public LocalMlRuntimeConfig build() {
            return new LocalMlRuntimeConfig(this);
        }
    }
}
