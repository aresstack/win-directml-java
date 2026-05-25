package com.aresstack.windirectml.runtime;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.EmbeddingModel;
import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;
import com.aresstack.windirectml.encoder.e5.E5Encoders;
import com.aresstack.windirectml.encoder.e5.E5Variant;
import com.aresstack.windirectml.encoder.minilm.CpuMiniLmEncoder;
import com.aresstack.windirectml.encoder.minilm.DirectMlMiniLmEncoder;
import com.aresstack.windirectml.encoder.reranker.BertCrossEncoderRerankers;
import com.aresstack.windirectml.encoder.reranker.RerankException;
import com.aresstack.windirectml.encoder.reranker.RerankRequest;
import com.aresstack.windirectml.encoder.reranker.RerankResult;
import com.aresstack.windirectml.encoder.reranker.Reranker;

import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * Public Java 21 ML runtime facade for local Windows ML use cases.
 * <p>
 * This is the single documented entry point for Java 21 applications
 * that want to use DirectML-accelerated embeddings and reranking
 * directly as a library, without the JSON-RPC sidecar process.
 * <p>
 * <b>Usage:</b>
 * <pre>{@code
 * try (WinDirectMlRuntime runtime = WinDirectMlRuntime.builder()
 *         .embeddingModelDir(Path.of("model/all-MiniLM-L6-v2"))
 *         .backend(Backend.AUTO)
 *         .build()) {
 *
 *     EmbeddingVector vec = runtime.embed(EmbeddingRequest.of("Hello world"));
 *     List<RerankResult> ranked = runtime.rerank(
 *             new RerankRequest("query", List.of("doc1", "doc2"), 2));
 * }
 * }</pre>
 * <p>
 * <b>Architecture:</b>
 * <pre>
 * Java 21 app
 *   → WinDirectMlRuntime (this class)
 *   → directml-encoder (EmbeddingModel, Reranker)
 *   → directml-windows-bindings (DirectML / D3D12)
 *
 * Java 8 app
 *   → directml-sidecar-client-java8
 *   → JSON-RPC
 *   → directml-sidecar (adapter over WinDirectMlRuntime)
 *   → same directml-encoder / directml-windows-bindings
 * </pre>
 *
 * @see Builder
 */
public final class WinDirectMlRuntime implements AutoCloseable {

    private final EmbeddingModel embeddingModel;
    private final Reranker reranker;
    private final String embeddingBackend;
    private final String rerankerBackend;

    private WinDirectMlRuntime(EmbeddingModel embeddingModel, String embeddingBackend,
                               Reranker reranker, String rerankerBackend) {
        this.embeddingModel = embeddingModel;
        this.embeddingBackend = embeddingBackend;
        this.reranker = reranker;
        this.rerankerBackend = rerankerBackend;
    }

    // ── Embedding API ────────────────────────────────────────────────────

    /**
     * @return {@code true} if the embedding model is loaded and ready.
     */
    public boolean isEmbeddingReady() {
        return embeddingModel != null && embeddingModel.isReady();
    }

    /**
     * Embed a single text input.
     *
     * @throws ModelReadinessException if no embedding model is configured or ready.
     * @throws EmbeddingException      if inference fails.
     */
    public EmbeddingVector embed(EmbeddingRequest request) throws EmbeddingException {
        Objects.requireNonNull(request, "request");
        requireEmbeddingReady();
        return embeddingModel.embed(request);
    }

    /**
     * Embed a batch of text inputs. Order is preserved.
     *
     * @throws ModelReadinessException if no embedding model is configured or ready.
     * @throws EmbeddingException      if inference fails.
     */
    public List<EmbeddingVector> embedBatch(List<EmbeddingRequest> requests) throws EmbeddingException {
        Objects.requireNonNull(requests, "requests");
        requireEmbeddingReady();
        return embeddingModel.embedBatch(requests);
    }

    /**
     * @return the dimensionality of the embedding model output, or {@code -1}
     *         if no embedding model is configured.
     */
    public int embeddingDimension() {
        return embeddingModel != null ? embeddingModel.dimension() : -1;
    }

    /**
     * @return the active embedding backend name ({@code "cpu"}, {@code "directml"},
     *         or {@code null} if no model is loaded).
     */
    public String embeddingBackend() {
        return embeddingBackend;
    }

    // ── Reranking API ────────────────────────────────────────────────────

    /**
     * @return {@code true} if the reranker model is loaded and ready.
     */
    public boolean isRerankerReady() {
        return reranker != null && reranker.isReady();
    }

    /**
     * Rerank documents against a query using a cross-encoder.
     *
     * @throws ModelReadinessException if no reranker model is configured or ready.
     * @throws RerankException         if inference fails.
     */
    public List<RerankResult> rerank(RerankRequest request) throws RerankException {
        Objects.requireNonNull(request, "request");
        requireRerankerReady();
        return reranker.rerank(request);
    }

    /**
     * @return the active reranker backend name ({@code "cpu"}, {@code "directml"},
     *         or {@code null} if no model is loaded).
     */
    public String rerankerBackend() {
        return rerankerBackend;
    }

    /**
     * @return the reranker model name, or {@code null} if no reranker is loaded.
     */
    public String rerankerModelName() {
        return reranker != null ? reranker.modelName() : null;
    }

    // ── Lifecycle ────────────────────────────────────────────────────────

    @Override
    public void close() {
        if (reranker != null) {
            try {
                reranker.close();
            } catch (Exception ignored) {}
        }
        if (embeddingModel instanceof AutoCloseable ac) {
            try {
                ac.close();
            } catch (Exception ignored) {}
        }
    }

    // ── Builder ──────────────────────────────────────────────────────────

    /**
     * Create a new builder for configuring the runtime.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Construct a runtime from pre-built model instances (for testing or
     * advanced wiring). Prefer {@link #builder()} for normal use.
     */
    public static WinDirectMlRuntime of(EmbeddingModel embeddingModel, String embeddingBackend,
                                        Reranker reranker, String rerankerBackend) {
        return new WinDirectMlRuntime(embeddingModel, embeddingBackend, reranker, rerankerBackend);
    }

    // ── Private helpers ──────────────────────────────────────────────────

    private void requireEmbeddingReady() {
        if (embeddingModel == null) {
            throw new ModelReadinessException("No embedding model configured. "
                    + "Use WinDirectMlRuntime.builder().embeddingModelDir(...) to configure one.");
        }
        if (!embeddingModel.isReady()) {
            throw new ModelReadinessException("Embedding model is not ready.");
        }
    }

    private void requireRerankerReady() {
        if (reranker == null) {
            throw new ModelReadinessException("No reranker model configured. "
                    + "Use WinDirectMlRuntime.builder().rerankerModelDir(...) to configure one.");
        }
        if (!reranker.isReady()) {
            throw new ModelReadinessException("Reranker model is not ready.");
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Builder
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Builder for {@link WinDirectMlRuntime}.
     * <p>
     * All configuration is optional – the runtime starts with only the
     * capabilities whose model directories are supplied.
     */
    public static final class Builder {

        private Path embeddingModelDir;
        private Path rerankerModelDir;
        private Backend backend = Backend.AUTO;
        private EmbeddingFamily embeddingFamily = EmbeddingFamily.MINILM;
        private E5Variant e5Variant = E5Variant.BASE_STS_EN_DE;

        Builder() {}

        /**
         * Path to the embedding model directory (must contain
         * {@code model.safetensors} and {@code tokenizer.json}).
         */
        public Builder embeddingModelDir(Path dir) {
            this.embeddingModelDir = dir;
            return this;
        }

        /**
         * Path to the reranker model directory (must contain
         * {@code model.safetensors}, {@code tokenizer.json}, and
         * {@code config.json}).
         */
        public Builder rerankerModelDir(Path dir) {
            this.rerankerModelDir = dir;
            return this;
        }

        /**
         * Execution provider selection: {@link Backend#AUTO} (default),
         * {@link Backend#CPU}, or {@link Backend#DIRECTML}.
         */
        public Builder backend(Backend backend) {
            this.backend = Objects.requireNonNull(backend, "backend");
            return this;
        }

        /**
         * Embedding model family (default: {@link EmbeddingFamily#MINILM}).
         */
        public Builder embeddingFamily(EmbeddingFamily family) {
            this.embeddingFamily = Objects.requireNonNull(family, "family");
            return this;
        }

        /**
         * E5 variant, only relevant when {@link #embeddingFamily(EmbeddingFamily)}
         * is {@link EmbeddingFamily#E5}. Default: {@link E5Variant#BASE_STS_EN_DE}.
         */
        public Builder e5Variant(E5Variant variant) {
            this.e5Variant = Objects.requireNonNull(variant, "variant");
            return this;
        }

        /**
         * Build the runtime. Loads model weights and initialises backends.
         *
         * @throws ModelReadinessException if a forced backend mode fails to initialise.
         */
        public WinDirectMlRuntime build() {
            EmbeddingModel embedModel = null;
            String embedBackend = null;
            Reranker rerankerModel = null;
            String rerankBackend = null;

            if (embeddingModelDir != null) {
                var selection = loadEmbeddingModel();
                embedModel = selection.model();
                embedBackend = selection.backend();
            }

            if (rerankerModelDir != null) {
                var selection = loadReranker();
                rerankerModel = selection.model();
                rerankBackend = selection.backend();
            }

            return new WinDirectMlRuntime(embedModel, embedBackend, rerankerModel, rerankBackend);
        }

        private record ModelSelection<T>(T model, String backend) {}

        private ModelSelection<EmbeddingModel> loadEmbeddingModel() {
            EncoderLoader cpuLoader = switch (embeddingFamily) {
                case MINILM -> CpuMiniLmEncoder::load;
                case E5 -> dir -> E5Encoders.loadCpu(dir, e5Variant);
            };
            EncoderLoader dmlLoader = switch (embeddingFamily) {
                case MINILM -> DirectMlMiniLmEncoder::load;
                case E5 -> dir -> E5Encoders.loadDirectMl(dir, e5Variant);
            };

            return switch (backend) {
                case CPU -> {
                    try {
                        yield new ModelSelection<>(cpuLoader.load(embeddingModelDir), "cpu");
                    } catch (Exception e) {
                        throw new ModelReadinessException(
                                "CPU embedding backend failed to load from "
                                        + embeddingModelDir + ": " + e.getMessage(), e);
                    }
                }
                case DIRECTML -> {
                    try {
                        yield new ModelSelection<>(dmlLoader.load(embeddingModelDir), "directml");
                    } catch (Exception e) {
                        throw new ModelReadinessException(
                                "DirectML embedding backend failed to load from "
                                        + embeddingModelDir + ": " + e.getMessage(), e);
                    }
                }
                case AUTO -> {
                    try {
                        yield new ModelSelection<>(dmlLoader.load(embeddingModelDir), "directml");
                    } catch (Exception primary) {
                        try {
                            yield new ModelSelection<>(cpuLoader.load(embeddingModelDir), "cpu");
                        } catch (Exception secondary) {
                            throw new ModelReadinessException(
                                    "Both DirectML and CPU embedding backends failed to load from "
                                            + embeddingModelDir + " (directml: " + primary.getMessage()
                                            + "; cpu: " + secondary.getMessage() + ")", secondary);
                        }
                    }
                }
            };
        }

        private ModelSelection<Reranker> loadReranker() {
            return switch (backend) {
                case CPU -> {
                    try {
                        yield new ModelSelection<>(
                                BertCrossEncoderRerankers.loadCpu(rerankerModelDir), "cpu");
                    } catch (Exception e) {
                        throw new ModelReadinessException(
                                "CPU reranker backend failed to load from "
                                        + rerankerModelDir + ": " + e.getMessage(), e);
                    }
                }
                case DIRECTML -> {
                    try {
                        yield new ModelSelection<>(
                                BertCrossEncoderRerankers.loadDirectMl(rerankerModelDir), "directml");
                    } catch (Exception e) {
                        throw new ModelReadinessException(
                                "DirectML reranker backend failed to load from "
                                        + rerankerModelDir + ": " + e.getMessage(), e);
                    }
                }
                case AUTO -> {
                    try {
                        yield new ModelSelection<>(
                                BertCrossEncoderRerankers.loadDirectMl(rerankerModelDir), "directml");
                    } catch (Exception primary) {
                        try {
                            yield new ModelSelection<>(
                                    BertCrossEncoderRerankers.loadCpu(rerankerModelDir), "cpu");
                        } catch (Exception secondary) {
                            throw new ModelReadinessException(
                                    "Both DirectML and CPU reranker backends failed to load from "
                                            + rerankerModelDir + " (directml: " + primary.getMessage()
                                            + "; cpu: " + secondary.getMessage() + ")", secondary);
                        }
                    }
                }
            };
        }

        @FunctionalInterface
        private interface EncoderLoader {
            EmbeddingModel load(Path modelDir) throws Exception;
        }
    }
}
