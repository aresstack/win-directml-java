package com.aresstack.windirectml.config.models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Shared, Java-8-compatible registry of the model checkpoints this
 * project knows about &mdash; including non-embedding models that
 * <b>must not</b> be accepted by the {@code embed} pipeline.
 *
 * <p>The registry exists to back two contracts:
 *
 * <ol>
 *   <li><b>Classification</b> &ndash; every {@code modelId} from the
 *       in-house model list maps to a {@link UseCase}
 *       ({@code embedding}, {@code reranker}, {@code summarizer},
 *       {@code decoder}, {@code unsupported}) and a {@link Status}
 *       ({@code shipped}, {@code experimental}, {@code planned},
 *       {@code unsupported}).</li>
 *   <li><b>Gating</b> &ndash; the sidecar's {@code embed} gate resolves
 *       {@code -Dembed.model} through this registry so that
 *       decoder / summarizer IDs are rejected with a clear,
 *       use-case-specific error rather than the generic
 *       "unknown family" fallback.</li>
 * </ol>
 *
 * <p>Lives in {@code directml-config} so the Java-8 workbench / client
 * can re-use the same classification (e.g. to filter the embedding
 * model dropdown to {@code useCase=embedding} entries only) without
 * pulling in the Java-21 sidecar code.
 *
 * <p>The fields mirror the schema requested by the company model-list
 * ticket: {@code modelId / useCase / provider / architecture /
 * tokenizerType / backendSupport / status / modelDirHints /
 * downloadScriptSupport / realModelTestStatus / notes}. Each embedding
 * entry additionally carries an optional {@link Entry#embedFamily()}
 * token that points at the implementation family (e.g. {@code minilm},
 * {@code e5}) &ndash; or {@code null} if no implementation exists yet.
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
     * copied; lists are never null (an empty list is used instead).
     *
     * <p>Accessor method names match the Java-14 record style
     * ({@code modelId()}, {@code useCase()}, &hellip;) so callers can
     * stay source-compatible across the Java-8 and Java-21 modules.
     */
    public static final class Entry {
        private final String modelId;
        private final UseCase useCase;
        private final String provider;
        private final String architecture;
        private final String tokenizerType;
        private final String backendSupport;
        private final Status status;
        private final List<String> modelDirHints;
        private final String downloadScriptSupport;
        private final String realModelTestStatus;
        private final String notes;
        private final String embedFamily;

        public Entry(String modelId,
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
            if (modelId == null || modelId.trim().isEmpty()) {
                throw new IllegalArgumentException("modelId must not be blank");
            }
            if (useCase == null) {
                throw new IllegalArgumentException("useCase must not be null");
            }
            if (status == null) {
                throw new IllegalArgumentException("status must not be null");
            }
            this.modelId = modelId;
            this.useCase = useCase;
            this.provider = provider;
            this.architecture = architecture;
            this.tokenizerType = tokenizerType;
            this.backendSupport = backendSupport;
            this.status = status;
            this.modelDirHints = modelDirHints == null
                    ? Collections.<String>emptyList()
                    : Collections.unmodifiableList(new ArrayList<String>(modelDirHints));
            this.downloadScriptSupport = downloadScriptSupport;
            this.realModelTestStatus = realModelTestStatus;
            this.notes = notes;
            this.embedFamily = embedFamily;
        }

        public String modelId() { return modelId; }
        public UseCase useCase() { return useCase; }
        public String provider() { return provider; }
        public String architecture() { return architecture; }
        public String tokenizerType() { return tokenizerType; }
        public String backendSupport() { return backendSupport; }
        public Status status() { return status; }
        public List<String> modelDirHints() { return modelDirHints; }
        public String downloadScriptSupport() { return downloadScriptSupport; }
        public String realModelTestStatus() { return realModelTestStatus; }
        public String notes() { return notes; }
        public String embedFamily() { return embedFamily; }

        public boolean isEmbedding() {
            return useCase == UseCase.EMBEDDING;
        }
    }

    // -- Entries ---------------------------------------------------------

    // The seven company-list models, in the order given by the ticket.
    private static final List<Entry> ENTRIES;
    private static final Map<String, Entry> BY_KEY;

    static {
        List<Entry> entries = new ArrayList<Entry>();
        entries.add(new Entry(
                "sentence-transformers/all-MiniLM-L6-v2",
                UseCase.EMBEDDING,
                "Huggingface/Sentence-Transformers",
                "MiniLM / BERT-style encoder",
                "WordPiece",
                "cpu, directml",
                Status.SHIPPED,
                Arrays.asList(
                        "model/all-MiniLM-L6-v2",
                        "model/sentence-transformers/all-MiniLM-L6-v2"),
                "scripts/download-minilm.ps1",
                "real-model tested (CPU + DirectML parity)",
                "default fast embedding model",
                "minilm"));
        entries.add(new Entry(
                "danielheinz/e5-base-sts-en-de",
                UseCase.EMBEDDING,
                "Daniel Heinz",
                "BERT-base (E5 fine-tune, en/de STS)",
                "WordPiece",
                "cpu, directml",
                Status.SHIPPED,
                Arrays.asList(
                        "model/e5-base-sts-en-de",
                        "model/danielheinz/e5-base-sts-en-de"),
                "scripts/download-e5.ps1",
                "real-model reference test present",
                "Use \"query: \" / \"passage: \" prefixes (E5 convention).",
                "e5"));
        entries.add(new Entry(
                "openai/gpt-oss-120b",
                UseCase.DECODER,
                "OpenAI",
                "decoder-only LLM",
                "n/a (not an embedding model)",
                "n/a",
                Status.UNSUPPORTED,
                Collections.<String>emptyList(),
                "none",
                "n/a",
                "Decoder model \u2013 belongs to a future text-generation ticket, "
                        + "not the embed endpoint.",
                null));
        entries.add(new Entry(
                "jinaai/jina-embeddings-v2-base-de",
                UseCase.EMBEDDING,
                "Jina AI",
                "Jina BERT v2 (custom; ALiBi positional bias)",
                "WordPiece (Jina-custom)",
                "planned",
                Status.PLANNED,
                Arrays.asList(
                        "model/jina-embeddings-v2-base-de",
                        "model/jinaai/jina-embeddings-v2-base-de"),
                "planned (no download-jina.ps1 yet)",
                "not yet tested \u2013 architecture analysis pending",
                "Jina v2 uses ALiBi instead of learned positional "
                        + "embeddings; not a drop-in for the current BERT core. "
                        + "Needs analysis before being marked experimental.",
                null));
        entries.add(new Entry(
                "casperhansen/llama-3.3-70b-instruct-awq",
                UseCase.DECODER,
                "Meta",
                "Llama 3.3 70B (AWQ-quantised) decoder-only LLM",
                "n/a (not an embedding model)",
                "n/a",
                Status.UNSUPPORTED,
                Collections.<String>emptyList(),
                "none",
                "n/a",
                "Decoder model \u2013 belongs to a future text-generation ticket, "
                        + "not the embed endpoint.",
                null));
        entries.add(new Entry(
                "intfloat/multilingual-e5-large-instruct",
                UseCase.EMBEDDING,
                "Microsoft / intfloat",
                "XLM-RoBERTa-large encoder (E5 instruct fine-tune)",
                "SentencePiece (XLM-R)",
                "planned",
                Status.PLANNED,
                Arrays.asList(
                        "model/multilingual-e5-large-instruct",
                        "model/intfloat/multilingual-e5-large-instruct"),
                "planned (current download-e5.ps1 only covers WordPiece variants)",
                "not yet tested \u2013 blocked on SentencePiece/XLM-R support",
                "NOT compatible with the current WordPiece-only E5 path. "
                        + "Requires SentencePiece tokenizer and XLM-RoBERTa weight "
                        + "naming \u2013 tracked separately. Uses an instruction-style "
                        + "prefix (\"Instruct: ...\\nQuery: ...\").",
                null));
        entries.add(new Entry(
                "ellamind/summarizer-v6-llama-v2",
                UseCase.SUMMARIZER,
                "Ellamind",
                "Llama-v2 summarizer fine-tune (decoder-only)",
                "n/a (not an embedding model)",
                "n/a",
                Status.UNSUPPORTED,
                Collections.<String>emptyList(),
                "none",
                "n/a",
                "Summarizer/decoder model \u2013 belongs to the summarize endpoint "
                        + "future work, not the embed endpoint.",
                null));
        ENTRIES = Collections.unmodifiableList(entries);

        Map<String, Entry> byKey = new LinkedHashMap<String, Entry>();
        for (Entry e : ENTRIES) {
            byKey.put(e.modelId().toLowerCase(Locale.ROOT), e);
        }
        BY_KEY = Collections.unmodifiableMap(byKey);
    }

    private EmbeddingModelRegistry() {
        // utility class
    }

    /** All known entries in declaration order. */
    public static List<Entry> entries() {
        return ENTRIES;
    }

    /**
     * Subset of {@link #entries()} restricted to a single
     * {@link UseCase}, in declaration order. Returned list is
     * unmodifiable.
     *
     * <p>The Java-8 workbench uses {@code entriesByUseCase(EMBEDDING)}
     * to populate its embedding model dropdown so decoder /
     * summarizer / unsupported IDs are never selectable for the
     * {@code embed} endpoint.
     */
    public static List<Entry> entriesByUseCase(UseCase useCase) {
        if (useCase == null) {
            throw new IllegalArgumentException("useCase must not be null");
        }
        List<Entry> out = new ArrayList<Entry>();
        for (Entry e : ENTRIES) {
            if (e.useCase() == useCase) {
                out.add(e);
            }
        }
        return Collections.unmodifiableList(out);
    }

    /**
     * Look up an entry by its full model ID (case-insensitive, e.g.
     * {@code "openai/gpt-oss-120b"}). Returns {@code null} for unknown
     * IDs or for short family aliases like {@code "minilm"}.
     *
     * <p>{@code null} (rather than {@code Optional}) keeps the API
     * Java-8 friendly without forcing every caller through
     * {@code Optional.ofNullable(...)}; the Java-21 sidecar wraps the
     * return value in an {@link java.util.Optional} at its own call
     * site.
     */
    public static Entry findByModelId(String modelId) {
        if (modelId == null) return null;
        String key = modelId.trim().toLowerCase(Locale.ROOT);
        if (key.isEmpty()) return null;
        return BY_KEY.get(key);
    }

    /**
     * Build the standard "X is not an embedding model" error message
     * used by the {@code embed} gate when a registered non-embedding
     * model ID is passed to {@code -Dembed.model}. The wording matches
     * the contract spelled out in the ticket so downstream tooling can
     * match on it.
     */
    public static String nonEmbeddingErrorMessage(Entry entry) {
        if (entry == null) {
            throw new IllegalArgumentException("entry must not be null");
        }
        switch (entry.useCase()) {
            case DECODER:
                return "Model " + entry.modelId()
                        + " is not an embedding model. Decoder models are not supported "
                        + "by the embed endpoint.";
            case SUMMARIZER:
                return "Model " + entry.modelId()
                        + " is not an embedding model. Summarizer models are not supported "
                        + "by the embed endpoint.";
            case RERANKER:
                return "Model " + entry.modelId()
                        + " is not an embedding model. Use the rerank endpoint instead.";
            case UNSUPPORTED:
                return "Model " + entry.modelId()
                        + " is not an embedding model and is not supported "
                        + "by the embed endpoint.";
            case EMBEDDING:
                throw new IllegalArgumentException(
                        "nonEmbeddingErrorMessage called for an embedding model: "
                                + entry.modelId());
            default:
                throw new IllegalStateException("unhandled useCase: " + entry.useCase());
        }
    }

    /**
     * Build the "embedding-class but not yet implemented" error
     * message for entries that are classified as embeddings but
     * carry {@code status=planned} or {@code status=experimental}
     * without an implementation hook (i.e. {@code embedFamily==null}).
     */
    public static String unimplementedEmbeddingErrorMessage(Entry entry) {
        if (entry == null) {
            throw new IllegalArgumentException("entry must not be null");
        }
        return "Model " + entry.modelId()
                + " is classified as an embedding model but has no runtime support "
                + "in this build (status=" + entry.status().token() + "). "
                + "See SUPPORTED_MODELS.md for the current classification.";
    }
}
