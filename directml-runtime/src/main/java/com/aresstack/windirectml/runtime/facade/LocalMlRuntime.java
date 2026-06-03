package com.aresstack.windirectml.runtime.facade;

import com.aresstack.windirectml.config.models.EmbeddingModelRegistry;
import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.EmbeddingModel;
import com.aresstack.windirectml.encoder.e5.E5Encoders;
import com.aresstack.windirectml.encoder.e5.E5Variant;
import com.aresstack.windirectml.encoder.minilm.CpuMiniLmEncoder;
import com.aresstack.windirectml.encoder.minilm.DirectMlMiniLmEncoder;
import com.aresstack.windirectml.encoder.reranker.BertCrossEncoderRerankers;
import com.aresstack.windirectml.encoder.reranker.Reranker;

import java.nio.file.Path;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;

/**
 * Java 21 facade for direct in-process use of the local ML runtime.
 * <p>
 * Java 21 applications use this class to obtain embeddings, batch
 * embeddings, and reranking <b>without</b> starting the JSON-RPC sidecar
 * process.
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * LocalMlRuntime runtime = LocalMlRuntime.create();
 *
 * // Embeddings
 * var embedCfg = EmbeddingModelConfig.miniLm(Path.of("model/all-MiniLM-L6-v2"));
 * try (var embeddings = runtime.loadEmbeddingModel(embedCfg)) {
 *     float[] vector = embeddings.embed("hello world");
 *     List<float[]> batch = embeddings.embedBatch(List.of("a", "b", "c"));
 * }
 *
 * // Reranking
 * var rerankCfg = new RerankerModelConfig(Path.of("model/cross-encoder-ms-marco-MiniLM-L-6-v2"));
 * try (var reranker = runtime.loadRerankerModel(rerankCfg)) {
 *     var results = reranker.rerank("search query", List.of("doc1", "doc2"));
 * }
 * }</pre>
 *
 * <h2>Supported model families</h2>
 * <ul>
 *   <li><b>minilm</b> – WordPiece BERT-style MiniLM encoders (shipped, fully supported).</li>
 *   <li><b>e5</b> – WordPiece E5 variants: {@code small-v2}, {@code base-v2},
 *       {@code large-v2} (shipped, fully supported). An explicit
 *       {@link com.aresstack.windirectml.encoder.e5.E5Variant E5Variant} must be
 *       specified via {@link EmbeddingModelConfig#e5(java.nio.file.Path, E5Variant, String)}.</li>
 * </ul>
 * <p>
 * XLM-R/SentencePiece-based E5 models (e.g. {@code danielheinz/e5-base-sts-en-de},
 * {@code intfloat/multilingual-e5-large-instruct}) are <b>planned but not yet
 * supported</b>. Attempting to load them will produce an explicit
 * {@link UnsupportedModelException}.
 *
 * @deprecated Prefer the higher-level
 * {@link com.aresstack.windirectml.runtime.api.MlRuntime} entry
 * point, which provides strongly-typed model identifiers and a
 * simpler configuration surface. This class remains fully
 * functional and will not be removed without a major version bump.
 */
@Deprecated
public final class LocalMlRuntime {

    private static final Set<String> SUPPORTED_FAMILIES = Set.of("minilm", "e5");

    private final Backend backend;

    private LocalMlRuntime(LocalMlRuntimeConfig config) {
        this.backend = config.backend();
    }

    /**
     * Create a runtime with the default configuration ({@link Backend#AUTO}).
     */
    public static LocalMlRuntime create() {
        return create(LocalMlRuntimeConfig.builder().build());
    }

    /**
     * Create a runtime with the given configuration.
     */
    public static LocalMlRuntime create(LocalMlRuntimeConfig config) {
        Objects.requireNonNull(config, "config");
        return new LocalMlRuntime(config);
    }

    /**
     * Load an embedding model from disk.
     * <p>
     * The model family must be one of the supported families ({@code "minilm"},
     * {@code "e5"}). Attempting to load an unsupported family produces an
     * {@link UnsupportedModelException}.
     *
     * @param config model configuration (directory, family, optional prefix).
     * @return a ready-to-use embedding model handle.
     * @throws EmbeddingException        if loading fails (missing files, bad weights, etc.).
     * @throws UnsupportedModelException if the model family is not supported.
     */
    public LocalEmbeddingModel loadEmbeddingModel(EmbeddingModelConfig config)
            throws EmbeddingException {
        Objects.requireNonNull(config, "config");
        String family = config.modelFamily().trim().toLowerCase(Locale.ROOT);

        validateEmbeddingFamily(family);

        // E5 requires an explicit variant selection
        if ("e5".equals(family) && config.e5Variant() == null) {
            throw new IllegalArgumentException(
                    "E5 family requires an explicit variant (e5Variant). "
                            + "Use EmbeddingModelConfig.e5(modelDir, E5Variant.BASE_V2, prefix) "
                            + "or another supported WordPiece variant (SMALL_V2, BASE_V2, LARGE_V2).");
        }

        EmbeddingModel model = switch (backend) {
            case CPU -> loadEmbeddingCpu(family, config);
            case DIRECTML, WARP -> loadEmbeddingDirectMl(family, config, nativeBackendName());
            case AUTO, HYBRID -> loadEmbeddingAuto(family, config);
        };

        return new LocalEmbeddingModel(model, config.prefix());
    }

    /**
     * Load a cross-encoder reranker model from disk.
     *
     * @param config reranker configuration (model directory).
     * @return a ready-to-use reranker model handle.
     * @throws EmbeddingException if loading fails.
     */
    public LocalRerankerModel loadRerankerModel(RerankerModelConfig config)
            throws EmbeddingException {
        Objects.requireNonNull(config, "config");
        Path modelDir = config.modelDir();

        Reranker reranker = switch (backend) {
            case CPU -> BertCrossEncoderRerankers.loadCpu(modelDir);
            case DIRECTML, WARP -> BertCrossEncoderRerankers.loadDirectMl(modelDir, nativeBackendName());
            // HYBRID is Qwen2-specific (GPU prefill + CPU decode); for embedding/
            // reranker encoders there is no decode loop, so HYBRID behaves like AUTO.
            case AUTO, HYBRID -> loadRerankerAuto(modelDir);
        };

        return new LocalRerankerModel(reranker);
    }

    /**
     * The configured backend for this runtime.
     */
    public Backend backend() {
        return backend;
    }

    // ── Private helpers ────────────────────────────────────────────────

    private void validateEmbeddingFamily(String family) {
        if (!SUPPORTED_FAMILIES.contains(family)) {
            // Check if this is a known-but-unsupported model from the registry
            for (EmbeddingModelRegistry.Entry entry : EmbeddingModelRegistry.entries()) {
                if (entry.embedFamily() != null
                        && entry.embedFamily().equalsIgnoreCase(family)) {
                    // Known family with implementation
                    return;
                }
            }
            // Check if the family maps to a planned/unsupported model
            String reason = resolveUnsupportedReason(family);
            throw new UnsupportedModelException(family, reason);
        }
    }

    private String resolveUnsupportedReason(String family) {
        // Look in the registry for models that might match this family name
        for (EmbeddingModelRegistry.Entry entry : EmbeddingModelRegistry.entries()) {
            String id = entry.modelId().toLowerCase(Locale.ROOT);
            if (id.contains(family)) {
                if (entry.status() == EmbeddingModelRegistry.Status.PLANNED) {
                    return "Model family '" + family + "' is planned but not yet supported. "
                            + entry.notes();
                }
                if (entry.status() == EmbeddingModelRegistry.Status.UNSUPPORTED) {
                    return "Model '" + entry.modelId() + "' is not supported: " + entry.notes();
                }
            }
        }
        return "Model family '" + family + "' is not recognized. "
                + "Supported families: " + SUPPORTED_FAMILIES;
    }

    private EmbeddingModel loadEmbeddingCpu(String family, EmbeddingModelConfig config)
            throws EmbeddingException {
        return switch (family) {
            case "minilm" -> CpuMiniLmEncoder.load(config.modelDir());
            case "e5" -> E5Encoders.loadCpu(config.modelDir(), config.e5Variant());
            default -> throw new UnsupportedModelException(family,
                    "No CPU loader for family: " + family);
        };
    }

    private EmbeddingModel loadEmbeddingDirectMl(String family, EmbeddingModelConfig config)
            throws EmbeddingException {
        return loadEmbeddingDirectMl(family, config, nativeBackendName());
    }

    private EmbeddingModel loadEmbeddingDirectMl(String family, EmbeddingModelConfig config, String nativeBackend)
            throws EmbeddingException {
        return switch (family) {
            case "minilm" -> DirectMlMiniLmEncoder.load(config.modelDir(), nativeBackend);
            case "e5" -> E5Encoders.loadDirectMl(config.modelDir(), config.e5Variant(), nativeBackend);
            default -> throw new UnsupportedModelException(family,
                    "No DirectML loader for family: " + family);
        };
    }

    private String nativeBackendName() {
        return backend == Backend.WARP ? "warp" : "directml";
    }

    private EmbeddingModel loadEmbeddingAuto(String family, EmbeddingModelConfig config)
            throws EmbeddingException {
        try {
            return loadEmbeddingDirectMl(family, config);
        } catch (Exception directMlEx) {
            try {
                return loadEmbeddingCpu(family, config);
            } catch (Exception cpuEx) {
                throw new EmbeddingException(
                        "Backend=auto: both DirectML and CPU failed for family '"
                                + family + "' (directml=" + directMlEx.getMessage()
                                + "; cpu=" + cpuEx.getMessage() + ")", cpuEx);
            }
        }
    }

    private Reranker loadRerankerAuto(Path modelDir) throws EmbeddingException {
        try {
            return BertCrossEncoderRerankers.loadDirectMl(modelDir);
        } catch (Exception directMlEx) {
            try {
                return BertCrossEncoderRerankers.loadCpu(modelDir);
            } catch (Exception cpuEx) {
                throw new EmbeddingException(
                        "Backend=auto: both DirectML and CPU failed for reranker"
                                + " (directml=" + directMlEx.getMessage()
                                + "; cpu=" + cpuEx.getMessage() + ")", cpuEx);
            }
        }
    }
}
