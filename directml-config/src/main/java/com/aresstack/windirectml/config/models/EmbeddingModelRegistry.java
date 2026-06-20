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

    /**
     * Intended consumer of the model in this project.
     */
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

    /**
     * Release-status of the runtime support for the model.
     */
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

        public String modelId() {
            return modelId;
        }

        public UseCase useCase() {
            return useCase;
        }

        public String provider() {
            return provider;
        }

        public String architecture() {
            return architecture;
        }

        public String tokenizerType() {
            return tokenizerType;
        }

        public String backendSupport() {
            return backendSupport;
        }

        public Status status() {
            return status;
        }

        public List<String> modelDirHints() {
            return modelDirHints;
        }

        public String downloadScriptSupport() {
            return downloadScriptSupport;
        }

        public String realModelTestStatus() {
            return realModelTestStatus;
        }

        public String notes() {
            return notes;
        }

        public String embedFamily() {
            return embedFamily;
        }

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
                "Workbench download tab",
                "real-model tested (CPU + DirectML parity)",
                "default fast embedding model",
                "minilm"));
        entries.add(new Entry(
                "danielheinz/e5-base-sts-en-de",
                UseCase.EMBEDDING,
                "Daniel Heinz",
                "XLM-RoBERTa / multilingual-E5 derivative",
                "SentencePiece / XLM-R",
                "planned",
                Status.PLANNED,
                Arrays.asList(
                        "model/e5-base-sts-en-de",
                        "model/danielheinz/e5-base-sts-en-de"),
                "Workbench download tab",
                "config mismatch with current WordPiece E5 runtime; XLM-R support pending",
                "Requires SentencePiece + XLM-R encoder path; belongs with "
                        + "multilingual-E5 analysis. Upstream checkpoint at "
                        + "huggingface.co/danielheinz/e5-base-sts-en-de hosts an "
                        + "XLMRobertaModel (vocab=250002, type_vocab_size=1), not the "
                        + "WordPiece BERT-base profile this runtime supports today.",
                null));
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
                "JinaBERT v2 (BERT-base, 12 layers / 768 hidden / 12 heads; "
                        + "ALiBi positional bias, GLU-style feed-forward, "
                        + "max sequence length 8192)",
                "WordPiece (Jina-custom tokenizer.json; bilingual de/en vocab)",
                "planned (no CPU or DirectML path)",
                Status.PLANNED,
                Arrays.asList(
                        "model/jina-embeddings-v2-base-de",
                        "model/jinaai/jina-embeddings-v2-base-de"),
                "not added yet \u2013 no Workbench download entry until at least a "
                        + "CPU real-model path exists (see SUPPORTED_MODELS.md \u00a71.1.2)",
                "not tested \u2013 no runtime path; no real-model test claimed",
                "Analysis (see SUPPORTED_MODELS.md \u00a71.1.2): mean pooling "
                        + "+ L2 normalisation; ALiBi (no learned positional "
                        + "embeddings) and GLU-style MLP make this NOT a drop-in "
                        + "for the current BERT/MiniLM/E5 core. Requires a "
                        + "Jina-specific attention path before it can move to "
                        + "experimental; remains planned in this release.",
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
                "XLM-RoBERTa-large encoder (24 layers, hidden=1024, "
                        + "type_vocab_size=1); E5 instruct fine-tune",
                "SentencePiece (XLM-R, sentencepiece.bpe.model)",
                "planned",
                Status.PLANNED,
                Arrays.asList(
                        "model/multilingual-e5-large-instruct",
                        "model/intfloat/multilingual-e5-large-instruct"),
                "planned (current Workbench download support only covers WordPiece variants)",
                "not yet tested \u2013 blocked on SentencePiece/XLM-R support",
                "NOT compatible with the current WordPiece-only E5 path. "
                        + "Architecture is XLM-RoBERTa-large (config "
                        + "model_type=xlm-roberta, weights under roberta.* with a "
                        + "single-segment type_vocab_size=1 and a learned "
                        + "positional embedding of 514). Tokenizer is SentencePiece "
                        + "BPE (\u2581 word-boundary marker, <s>/</s>/<pad>/<unk> "
                        + "specials). Pooling is mean + L2, identical to the "
                        + "existing E5 core. Input format is the E5-instruct "
                        + "prefix \"Instruct: {task}\\nQuery: {query}\" for queries; "
                        + "passages are embedded without a prefix. See "
                        + "SUPPORTED_MODELS.md section 1.1.1 for the full analysis. "
                        + "Status stays planned until both SentencePiece and an "
                        + "XLM-RoBERTa encoder path land and a CPU real-model test "
                        + "is added.",
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
        entries.add(new Entry(
                "microsoft/Phi-3-mini-4k-instruct-onnx",
                UseCase.SUMMARIZER,
                "Microsoft",
                "Phi-3 Mini 4K Instruct (decoder-only, native Java/DirectML)",
                "SentencePiece (Phi-3 tokenizer.json)",
                "cpu, directml",
                Status.EXPERIMENTAL,
                Arrays.asList(
                        "model/phi-3-mini-4k-instruct-onnx",
                        "model/microsoft/Phi-3-mini-4k-instruct-onnx"),
                "Workbench download tab",
                "manual smoke test (see WINDOWS-SMOKE-RUN.md)",
                "Sidecar summarizer backend. Runs the native Java/DirectML Phi-3 decoder "
                        + "(no Python/ONNX Runtime) over the ONNX-format weights. Requires ~2.3 GB "
                        + "disk for the int4-quantised weights. CPU fallback supported; DirectML "
                        + "acceleration available on compatible GPUs. Not executable in the "
                        + "Workbench (no wdmlpack compiler) – see PHI3-PRODUCT-AUDIT-1.",
                null));
        entries.add(new Entry(
                "microsoft/Phi-3.5-mini-instruct-onnx",
                UseCase.SUMMARIZER,
                "Microsoft",
                "Phi-3.5 Mini Instruct (decoder-only, native Java/DirectML)",
                "SentencePiece (Phi-3.5 tokenizer.json)",
                "cpu, directml",
                Status.PLANNED,
                Arrays.asList(
                        "model/phi-3.5-mini-instruct-onnx",
                        "model/microsoft/Phi-3.5-mini-instruct-onnx"),
                "planned (download support not yet added)",
                "not yet tested \u2013 blocked on Phi-3.5 ONNX-format weights availability",
                "Successor to Phi-3 Mini with improved instruction following. "
                        + "Same architecture class as Phi-3 Mini 4K; expected to work "
                        + "with the same native Java/DirectML decoder path once the official "
                        + "weights are published. Tracked as a follow-up to the "
                        + "Phi-3 summarizer implementation.",
                null));
        entries.add(new Entry(
                "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                UseCase.DECODER,
                "Alibaba / Qwen",
                "Qwen2.5-Coder 0.5B decoder-only (24 layers, hidden=896, heads=14); "
                        + "ONNX INT4 AWQ block-128",
                "BPE (HuggingFace fast-tokenizer format; ChatML template)",
                "planned (CPU-first, DirectML later)",
                Status.PLANNED,
                Arrays.asList(
                        "model/qwen2.5-coder-0.5b-directml-int4"),
                "Workbench download tab (disabled pending source verification)",
                "not yet tested \u2013 ONNX source TBD/research",
                "Qwen2.5-Coder 0.5B Instruct. ONNX source is TBD/research; "
                        + "the download button is disabled until source verification "
                        + "completes. Runtime depends on #99. See "
                        + "docs/decision-qwen-artifact-format.md.",
                null));
        entries.add(new Entry(
                "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                UseCase.DECODER,
                "Alibaba / Qwen",
                "Qwen2.5-Coder 1.5B decoder-only (28 layers, hidden=1536, heads=12); "
                        + "ONNX INT4 AWQ block-128",
                "BPE (HuggingFace fast-tokenizer format; ChatML template)",
                "planned (CPU-first, DirectML later)",
                Status.PLANNED,
                Collections.<String>emptyList(),
                "none (planned)",
                "not yet tested \u2013 blocked on 0.5B runtime verification",
                "Qwen2.5-Coder 1.5B Instruct scale-up candidate. Same format "
                        + "as 0.5B, larger weights (~1 GB INT4). Not enabled until "
                        + "0.5B runtime smoke test passes.",
                null));
        entries.add(new Entry(
                "Qwen/Qwen2.5-Coder-3B-Instruct",
                UseCase.DECODER,
                "Alibaba / Qwen",
                "Qwen2.5-Coder 3B decoder-only (36 layers, hidden=2048, heads=16); "
                        + "ONNX INT4 AWQ block-128",
                "BPE (HuggingFace fast-tokenizer format; ChatML template)",
                "planned (CPU-first, DirectML later)",
                Status.PLANNED,
                Collections.<String>emptyList(),
                "none (planned)",
                "not yet tested \u2013 blocked on 0.5B runtime verification",
                "Qwen2.5-Coder 3B Instruct scale-up candidate. Same format "
                        + "as 0.5B, larger weights (~2 GB INT4). May require chunked "
                        + "mmap. Not enabled until 0.5B runtime smoke test passes.",
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

    /**
     * All known entries in declaration order.
     */
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
