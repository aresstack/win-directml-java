package com.aresstack.windirectml.sidecar;

import com.aresstack.windirectml.encoder.EmbeddingModel;
import com.aresstack.windirectml.encoder.e5.E5Variant;
import com.aresstack.windirectml.runtime.facade.Backend;
import com.aresstack.windirectml.runtime.facade.EmbeddingModelConfig;
import com.aresstack.windirectml.runtime.facade.LocalEmbeddingModel;
import com.aresstack.windirectml.runtime.facade.LocalMlRuntime;
import com.aresstack.windirectml.runtime.facade.LocalMlRuntimeConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.Locale;

/**
 * Selects the embedding backend using the public {@link LocalMlRuntime} facade.
 * <p>
 * Three modes:
 * <ul>
 *   <li>{@code cpu} – force CPU; fail visibly if loading fails.</li>
 *   <li>{@code directml} – force DirectML; fail visibly if unavailable.</li>
 *   <li>{@code auto} (default) – try DirectML first via the runtime, fall
 *       back to CPU on failure and record a warning.</li>
 * </ul>
 */
public final class EmbeddingBackendSelector {

    private static final Logger log = LoggerFactory.getLogger(EmbeddingBackendSelector.class);

    /**
     * Backend mode derived from {@code -Dembed.backend}.
     */
    public enum Mode {
        CPU, DIRECTML, AUTO;

        /**
         * Parse a {@code -Dembed.backend} property value.
         * {@code null}/blank → {@link #AUTO}. Unknown values throw
         * {@link IllegalArgumentException}.
         */
        public static Mode parse(String raw) {
            if (raw == null || raw.isBlank()) return AUTO;
            return switch (raw.trim().toLowerCase(Locale.ROOT)) {
                case "cpu" -> CPU;
                case "directml", "dml" -> DIRECTML;
                case "auto" -> AUTO;
                default -> throw new IllegalArgumentException(
                        "Unknown embed.backend: '" + raw
                                + "' (expected one of: cpu, directml, auto)");
            };
        }

        /**
         * Stable lowercase token used in health/log output.
         */
        public String token() {
            return switch (this) {
                case CPU -> "cpu";
                case DIRECTML -> "directml";
                case AUTO -> "auto";
            };
        }
    }

    /**
     * Functional interface for loading an embedding model from a directory.
     * Retained for backward compatibility with tests that supply custom loaders.
     */
    @FunctionalInterface
    public interface EncoderLoader {
        EmbeddingModel load(Path modelDir) throws Exception;
    }

    /**
     * Result of a backend selection.
     */
    public record Selection(EmbeddingModel model,
                            String backend,
                            String warning,
                            boolean fallback) {
    }

    private final EncoderLoader cpuLoader;
    private final EncoderLoader directmlLoader;

    public EmbeddingBackendSelector(EncoderLoader cpuLoader, EncoderLoader directmlLoader) {
        this.cpuLoader = cpuLoader;
        this.directmlLoader = directmlLoader;
    }

    /**
     * Select and load the embedding model using custom loaders (test path).
     */
    public Selection select(Mode mode, Path modelDir) {
        switch (mode) {
            case CPU: {
                try {
                    EmbeddingModel m = cpuLoader.load(modelDir);
                    log.info("embed.backend=cpu: CPU encoder loaded from {}", modelDir);
                    return new Selection(m, "cpu", null, false);
                } catch (Exception e) {
                    throw new IllegalStateException(
                            "embed.backend=cpu requested but CPU encoder failed to load: "
                                    + e.getMessage(), e);
                }
            }
            case DIRECTML: {
                try {
                    EmbeddingModel m = directmlLoader.load(modelDir);
                    log.info("embed.backend=directml: DirectML encoder loaded from {}", modelDir);
                    return new Selection(m, "directml", null, false);
                } catch (Exception e) {
                    throw new IllegalStateException(
                            "embed.backend=directml requested but DirectML encoder failed to load: "
                                    + e.getMessage(), e);
                }
            }
            case AUTO: {
                try {
                    EmbeddingModel m = directmlLoader.load(modelDir);
                    log.info("embed.backend=auto: DirectML encoder loaded from {}", modelDir);
                    return new Selection(m, "directml", null, false);
                } catch (Exception primary) {
                    String warn = "embed.backend=auto: DirectML unavailable, falling back to CPU – "
                            + primary.getMessage();
                    log.warn(warn);
                    try {
                        EmbeddingModel m = cpuLoader.load(modelDir);
                        return new Selection(m, "cpu", warn, true);
                    } catch (Exception secondary) {
                        throw new IllegalStateException(
                                "embed.backend=auto: both DirectML and CPU encoders failed (directml="
                                        + primary.getMessage() + "; cpu=" + secondary.getMessage() + ")",
                                secondary);
                    }
                }
            }
            default:
                throw new IllegalStateException("unreachable mode: " + mode);
        }
    }

    /**
     * Select and load the embedding model using the public {@link LocalMlRuntime}
     * facade. This is the production path used by the sidecar entry point.
     *
     * @param mode       backend mode (cpu/directml/auto)
     * @param modelDir   model directory
     * @param family     embedding family ("minilm" or "e5")
     * @param e5Variant  E5 variant (required for e5 family, null otherwise)
     * @param prefix     optional E5 prefix
     * @return selection with loaded model, backend name, and fallback info
     */
    public static Selection selectViaRuntime(Mode mode, Path modelDir, String family,
                                             E5Variant e5Variant, String prefix) {
        switch (mode) {
            case CPU: {
                try {
                    LocalMlRuntime runtime = LocalMlRuntime.create(
                            LocalMlRuntimeConfig.builder().backend(Backend.CPU).build());
                    EmbeddingModelConfig config = buildEmbedConfig(modelDir, family, e5Variant, prefix);
                    LocalEmbeddingModel loaded = runtime.loadEmbeddingModel(config);
                    EmbeddingModel m = loaded.unwrapModel();
                    log.info("embed.backend=cpu: loaded via runtime from {}", modelDir);
                    return new Selection(m, "cpu", null, false);
                } catch (Exception e) {
                    throw new IllegalStateException(
                            "embed.backend=cpu requested but CPU encoder failed to load: "
                                    + e.getMessage(), e);
                }
            }
            case DIRECTML: {
                try {
                    LocalMlRuntime runtime = LocalMlRuntime.create(
                            LocalMlRuntimeConfig.builder().backend(Backend.DIRECTML).build());
                    EmbeddingModelConfig config = buildEmbedConfig(modelDir, family, e5Variant, prefix);
                    LocalEmbeddingModel loaded = runtime.loadEmbeddingModel(config);
                    EmbeddingModel m = loaded.unwrapModel();
                    log.info("embed.backend=directml: loaded via runtime from {}", modelDir);
                    return new Selection(m, "directml", null, false);
                } catch (Exception e) {
                    throw new IllegalStateException(
                            "embed.backend=directml requested but DirectML encoder failed to load: "
                                    + e.getMessage(), e);
                }
            }
            case AUTO: {
                try {
                    LocalMlRuntime runtime = LocalMlRuntime.create(
                            LocalMlRuntimeConfig.builder().backend(Backend.DIRECTML).build());
                    EmbeddingModelConfig config = buildEmbedConfig(modelDir, family, e5Variant, prefix);
                    LocalEmbeddingModel loaded = runtime.loadEmbeddingModel(config);
                    EmbeddingModel m = loaded.unwrapModel();
                    log.info("embed.backend=auto: DirectML loaded via runtime from {}", modelDir);
                    return new Selection(m, "directml", null, false);
                } catch (Exception primary) {
                    String warn = "embed.backend=auto: DirectML unavailable, falling back to CPU – "
                            + primary.getMessage();
                    log.warn(warn);
                    try {
                        LocalMlRuntime runtime = LocalMlRuntime.create(
                                LocalMlRuntimeConfig.builder().backend(Backend.CPU).build());
                        EmbeddingModelConfig config = buildEmbedConfig(modelDir, family, e5Variant, prefix);
                        LocalEmbeddingModel loaded = runtime.loadEmbeddingModel(config);
                        EmbeddingModel m = loaded.unwrapModel();
                        return new Selection(m, "cpu", warn, true);
                    } catch (Exception secondary) {
                        throw new IllegalStateException(
                                "embed.backend=auto: both DirectML and CPU failed (directml="
                                        + primary.getMessage() + "; cpu=" + secondary.getMessage() + ")",
                                secondary);
                    }
                }
            }
            default:
                throw new IllegalStateException("unreachable mode: " + mode);
        }
    }

    private static EmbeddingModelConfig buildEmbedConfig(Path modelDir, String family,
                                                         E5Variant e5Variant, String prefix) {
        if ("e5".equals(family)) {
            return EmbeddingModelConfig.e5(modelDir, e5Variant, prefix);
        }
        return EmbeddingModelConfig.miniLm(modelDir);
    }
}

