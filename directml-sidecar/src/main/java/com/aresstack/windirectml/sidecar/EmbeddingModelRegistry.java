package com.aresstack.windirectml.sidecar;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Central, single-source-of-truth registry for the model checkpoints
 * that the project knows about – including non-embedding models that
 * <b>must not</b> be accepted by the {@code embed} pipeline.
 *
 * <p>The registry exists to back two contracts:
 *
 * <ol>
 *   <li><b>Classification</b> – every modelId from the in-house model
 *       list maps to a {@link UseCase} ({@code embedding}, {@code
 *       reranker}, {@code summarizer}, {@code decoder},
 *       {@code unsupported}) and a {@link Status}
 *       ({@code shipped}, {@code experimental}, {@code planned},
 *       {@code unsupported}).</li>
 *   <li><b>Gating</b> – {@link
 *       DirectMlPhi3Sidecar#embedFamily(String)} resolves
 *       {@code -Dembed.model} through this registry so that
 *       decoder / summarizer IDs are rejected with a clear,
 *       use-case-specific error rather than the generic
 *       "unknown family" fallback.</li>
 * </ol>
 *
 * <p>The fields mirror the schema requested by the company model-list
 * ticket: {@code modelId / useCase / provider / architecture /
 * tokenizerType / backendSupport / status / modelDirHints /
 * downloadScriptSupport / realModelTestStatus / notes}. Additionally
 * each embedding entry carries an optional {@link Entry#embedFamily}
 * token that points at the implementation family (e.g. {@code minilm},
 * {@code e5}) – or {@code null} if no implementation exists yet.
 */
public final class EmbeddingModelRegistry {

    /** Intended consumer of the model in this project. */
    public enum UseCase {
        EMBEDDING,
        RERANKER,
        SUMMARIZER,
        DECODER,
        UNSUPPORTED;

        public String token() {
            return name().toLowerCase(Locale.ROOT);
        }
    }

    /** Release-status of the runtime support for the model. */
    public enum Status {
        SHIPPED,
        EXPERIMENTAL,
        PLANNED,
        UNSUPPORTED;

        public String token() {
            return name().toLowerCase(Locale.ROOT);
        }
    }

    /**
     * Immutable registry record. {@code modelDirHints} is defensively
     * copied; lists are never null (use empty list instead).
     */
    public record Entry(
            String modelId,
            UseCase useCase,
            String provider,
            String architecture,
            String tokenizerType,
            String backendSupport,
            Status status,
            List<String> modelDirHints,
            String downloadScriptSupport,
            String realModelTestStatus,
            String notes,
            String embedFamily) {

        public Entry {
            if (modelId == null || modelId.isBlank()) {
                throw new IllegalArgumentException("modelId must not be blank");
            }
            if (useCase == null) {
                throw new IllegalArgumentException("useCase must not be null");
            }
            if (status == null) {
                throw new IllegalArgumentException("status must not be null");
            }
            modelDirHints = modelDirHints == null ? List.of() : List.copyOf(modelDirHints);
        }

        public boolean isEmbedding() {
            return useCase == UseCase.EMBEDDING;
        }
    }

    // ── Entries ─────────────────────────────────────────────────────────

    // The seven company-list models, in the order given by the ticket.
    private static final List<Entry> ENTRIES = List.of(
            new Entry(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    UseCase.EMBEDDING,
                    "Huggingface/Sentence-Transformers",
                    "MiniLM / BERT-style encoder",
                    "WordPiece",
                    "cpu, directml",
                    Status.SHIPPED,
                    List.of("model/all-MiniLM-L6-v2",
                            "model/sentence-transformers/all-MiniLM-L6-v2"),
                    "scripts/download-minilm.ps1",
                    "real-model tested (CPU + DirectML parity)",
                    "default fast embedding model",
                    "minilm"),
            new Entry(
                    "danielheinz/e5-base-sts-en-de",
                    UseCase.EMBEDDING,
                    "Daniel Heinz",
                    "BERT-base (E5 fine-tune, en/de STS)",
                    "WordPiece",
                    "cpu, directml",
                    Status.SHIPPED,
                    List.of("model/e5-base-sts-en-de",
                            "model/danielheinz/e5-base-sts-en-de"),
                    "scripts/download-e5.ps1",
                    "real-model reference test present",
                    "Use \"query: \" / \"passage: \" prefixes (E5 convention).",
                    "e5"),
            new Entry(
                    "openai/gpt-oss-120b",
                    UseCase.DECODER,
                    "OpenAI",
                    "decoder-only LLM",
                    "n/a (not an embedding model)",
                    "n/a",
                    Status.UNSUPPORTED,
                    List.of(),
                    "none",
                    "n/a",
                    "Decoder model – belongs to a future text-generation ticket, " +
                            "not the embed endpoint.",
                    null),
            new Entry(
                    "jinaai/jina-embeddings-v2-base-de",
                    UseCase.EMBEDDING,
                    "Jina AI",
                    "Jina BERT v2 (custom; ALiBi positional bias)",
                    "WordPiece (Jina-custom)",
                    "planned",
                    Status.PLANNED,
                    List.of("model/jina-embeddings-v2-base-de",
                            "model/jinaai/jina-embeddings-v2-base-de"),
                    "planned (no download-jina.ps1 yet)",
                    "not yet tested – architecture analysis pending",
                    "Jina v2 uses ALiBi instead of learned positional " +
                            "embeddings; not a drop-in for the current BERT core. " +
                            "Needs analysis before being marked experimental.",
                    null),
            new Entry(
                    "casperhansen/llama-3.3-70b-instruct-awq",
                    UseCase.DECODER,
                    "Meta",
                    "Llama 3.3 70B (AWQ-quantised) decoder-only LLM",
                    "n/a (not an embedding model)",
                    "n/a",
                    Status.UNSUPPORTED,
                    List.of(),
                    "none",
                    "n/a",
                    "Decoder model – belongs to a future text-generation ticket, " +
                            "not the embed endpoint.",
                    null),
            new Entry(
                    "intfloat/multilingual-e5-large-instruct",
                    UseCase.EMBEDDING,
                    "Microsoft / intfloat",
                    "XLM-RoBERTa-large encoder (E5 instruct fine-tune)",
                    "SentencePiece (XLM-R)",
                    "planned",
                    Status.PLANNED,
                    List.of("model/multilingual-e5-large-instruct",
                            "model/intfloat/multilingual-e5-large-instruct"),
                    "planned (current download-e5.ps1 only covers WordPiece variants)",
                    "not yet tested – blocked on SentencePiece/XLM-R support",
                    "NOT compatible with the current WordPiece-only E5 path. " +
                            "Requires SentencePiece tokenizer and XLM-RoBERTa weight " +
                            "naming – tracked separately. Uses an instruction-style " +
                            "prefix (\"Instruct: ...\\nQuery: ...\").",
                    null),
            new Entry(
                    "ellamind/summarizer-v6-llama-v2",
                    UseCase.SUMMARIZER,
                    "Ellamind",
                    "Llama-v2 summarizer fine-tune (decoder-only)",
                    "n/a (not an embedding model)",
                    "n/a",
                    Status.UNSUPPORTED,
                    List.of(),
                    "none",
                    "n/a",
                    "Summarizer/decoder model – belongs to the summarize endpoint " +
                            "future work, not the embed endpoint.",
                    null)
    );

    private static final Map<String, Entry> BY_KEY = buildIndex(ENTRIES);

    private static Map<String, Entry> buildIndex(List<Entry> entries) {
        Map<String, Entry> map = new LinkedHashMap<>();
        for (Entry e : entries) {
            map.put(e.modelId().toLowerCase(Locale.ROOT), e);
        }
        return Map.copyOf(map);
    }

    private EmbeddingModelRegistry() {
        // utility class
    }

    /** All known entries in declaration order. */
    public static List<Entry> entries() {
        return ENTRIES;
    }

    /**
     * Look up an entry by its full model ID (case-insensitive, e.g.
     * {@code "openai/gpt-oss-120b"}). Returns empty for unknown IDs
     * or for short family aliases like {@code "minilm"}.
     */
    public static Optional<Entry> findByModelId(String modelId) {
        if (modelId == null) return Optional.empty();
        String key = modelId.trim().toLowerCase(Locale.ROOT);
        if (key.isEmpty()) return Optional.empty();
        return Optional.ofNullable(BY_KEY.get(key));
    }

    /**
     * Build the standard "X is not an embedding model" error message
     * used by the {@code embed} gate when a registered non-embedding
     * model ID is passed to {@code -Dembed.model}. The wording matches
     * the contract spelled out in the ticket so downstream tooling can
     * match on it.
     */
    public static String nonEmbeddingErrorMessage(Entry entry) {
        return switch (entry.useCase()) {
            case DECODER -> "Model " + entry.modelId()
                    + " is not an embedding model. Decoder models are not supported "
                    + "by the embed endpoint.";
            case SUMMARIZER -> "Model " + entry.modelId()
                    + " is not an embedding model. Summarizer models are not supported "
                    + "by the embed endpoint.";
            case RERANKER -> "Model " + entry.modelId()
                    + " is not an embedding model. Use the rerank endpoint instead.";
            case UNSUPPORTED -> "Model " + entry.modelId()
                    + " is not an embedding model and is not supported "
                    + "by the embed endpoint.";
            case EMBEDDING -> throw new IllegalArgumentException(
                    "nonEmbeddingErrorMessage called for an embedding model: "
                            + entry.modelId());
        };
    }

    /**
     * Build the "embedding-class but not yet implemented" error
     * message for entries that are classified as embeddings but
     * carry {@code status=planned} or {@code status=experimental}
     * without an implementation hook (i.e. {@code embedFamily==null}).
     */
    public static String unimplementedEmbeddingErrorMessage(Entry entry) {
        return "Model " + entry.modelId()
                + " is classified as an embedding model but has no runtime support "
                + "in this build (status=" + entry.status().token() + "). "
                + "See SUPPORTED_MODELS.md for the current classification.";
    }
}
