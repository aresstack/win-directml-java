package com.aresstack.windirectml.inference.t5;

import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Describes the curated T5/CodeT5 model family without creating runtimes or services.
 */
public final class T5ModelFamily {
    public static final String ID = "t5";
    public static final String DISPLAY_NAME = "T5 / CodeT5";
    public static final String ARCHITECTURE = "encoder-decoder";

    public String id() {
        return ID;
    }

    public String displayName() {
        return DISPLAY_NAME;
    }

    public boolean supports(T5Config config) {
        Objects.requireNonNull(config, "config");
        return config.encoderDecoder() && hasSupportedModelType(config) && hasSupportedArchitecture(config);
    }

    public T5Architecture architecture(T5Config config) {
        if (!supports(config)) {
            throw new IllegalArgumentException("Unsupported T5/CodeT5 configuration");
        }
        return config.architecture();
    }

    public T5SpecialTokens specialTokens(T5Config config) {
        if (!supports(config)) {
            throw new IllegalArgumentException("Unsupported T5/CodeT5 configuration");
        }
        return config.specialTokens();
    }

    public T5PackageMetadata packageMetadata(T5Config config) {
        if (!supports(config)) {
            throw new IllegalArgumentException("Unsupported T5/CodeT5 configuration");
        }
        return T5PackageMetadata.from(config);
    }

    private boolean hasSupportedModelType(T5Config config) {
        return config.modelType() == null || config.modelType().isBlank() || "t5".equals(config.modelType());
    }

    private boolean hasSupportedArchitecture(T5Config config) {
        List<String> architectures = config.architectures();
        if (architectures == null || architectures.isEmpty()) {
            return true;
        }
        for (String architecture : architectures) {
            String normalized = architecture == null ? "" : architecture.toLowerCase(Locale.ROOT);
            if (normalized.contains("t5") || normalized.contains("codet5")) {
                return true;
            }
        }
        return false;
    }
}
