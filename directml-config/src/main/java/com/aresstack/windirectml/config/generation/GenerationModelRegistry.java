package com.aresstack.windirectml.config.generation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Registry of text-generation model checkpoints known to this project.
 *
 * <p>Complements {@link com.aresstack.windirectml.config.models.EmbeddingModelRegistry}
 * which classifies embeddings and rerankers. This registry covers models that
 * produce <em>text output</em> via autoregressive decoding or sequence-to-sequence
 * generation.
 *
 * <p>The registry supports the following model families:
 * <ul>
 *   <li><b>Decoder-only / Causal LM</b> &ndash; Phi-3, Qwen, Llama, GPT-style</li>
 *   <li><b>Seq2Seq</b> &ndash; T5, BART (future; see issue #95)</li>
 * </ul>
 *
 * <p>Use-case adapters (summarizer, code-explanation) are <em>not</em> separate
 * model entries; they are application-layer wrappers around a generation model.
 *
 * <p>Java-8 compatible. Lives in {@code directml-config} so the workbench can
 * query generation-capable models without depending on Java-21 inference code.
 */
public final class GenerationModelRegistry {

    /** Architecture family of the generation model. */
    public enum Architecture {
        /** Autoregressive decoder-only (Phi-3, Qwen, Llama, GPT). */
        CAUSAL_LM,
        /** Encoder-decoder / sequence-to-sequence (T5, BART). */
        SEQ2SEQ;

        public String token() {
            return name().toLowerCase(Locale.ROOT);
        }
    }

    /** Release-status of the runtime support for the model. */
    public enum Status {
        /** Fully supported and tested. */
        SHIPPED,
        /** Partially working, may have limitations. */
        EXPERIMENTAL,
        /** Metadata only; runtime not yet implemented. */
        PLANNED,
        /** Not supported by this project. */
        UNSUPPORTED;

        public String token() {
            return name().toLowerCase(Locale.ROOT);
        }
    }

    /**
     * Immutable registry entry for a generation model.
     *
     * <p>Accessor method names match the Java-14 record style for
     * forward compatibility.
     */
    public static final class Entry {
        private final String modelId;
        private final Architecture architecture;
        private final String provider;
        private final String parameterSize;
        private final ChatTemplate chatTemplate;
        private final Status status;
        private final List<String> modelDirHints;
        private final String notes;

        public Entry(String modelId,
                     Architecture architecture,
                     String provider,
                     String parameterSize,
                     ChatTemplate chatTemplate,
                     Status status,
                     List<String> modelDirHints,
                     String notes) {
            String normalizedModelId = modelId == null ? null : modelId.trim();
            if (normalizedModelId == null || normalizedModelId.isEmpty()) {
                throw new IllegalArgumentException("modelId must not be blank");
            }
            if (architecture == null) {
                throw new IllegalArgumentException("architecture must not be null");
            }
            if (status == null) {
                throw new IllegalArgumentException("status must not be null");
            }
            this.modelId = normalizedModelId;
            this.architecture = architecture;
            this.provider = provider;
            this.parameterSize = parameterSize;
            this.chatTemplate = chatTemplate != null ? chatTemplate : ChatTemplate.UNKNOWN;
            this.status = status;
            this.modelDirHints = modelDirHints == null
                    ? Collections.<String>emptyList()
                    : Collections.unmodifiableList(new ArrayList<String>(modelDirHints));
            this.notes = notes;
        }

        public String modelId() { return modelId; }
        public Architecture architecture() { return architecture; }
        public String provider() { return provider; }
        public String parameterSize() { return parameterSize; }
        public ChatTemplate chatTemplate() { return chatTemplate; }
        public Status status() { return status; }
        public List<String> modelDirHints() { return modelDirHints; }
        public String notes() { return notes; }

        /** Whether the model is a causal (decoder-only) language model. */
        public boolean isCausalLM() {
            return architecture == Architecture.CAUSAL_LM;
        }

        /** Whether the model has an active runtime (shipped or experimental). */
        public boolean isRunnable() {
            return status == Status.SHIPPED || status == Status.EXPERIMENTAL;
        }
    }

    // -- Entries ---------------------------------------------------------

    private static final List<Entry> ENTRIES;
    private static final Map<String, Entry> BY_KEY;

    static {
        List<Entry> entries = new ArrayList<Entry>();

        // --- Phi-3: experimental decoder/summarizer implementation ---
        entries.add(new Entry(
                "microsoft/Phi-3-mini-4k-instruct-onnx",
                Architecture.CAUSAL_LM,
                "Microsoft",
                "3.8B",
                ChatTemplate.PHI3,
                Status.EXPERIMENTAL,
                Arrays.asList(
                        "model/phi-3-mini-4k-instruct-onnx",
                        "model/microsoft/Phi-3-mini-4k-instruct-onnx"),
                "First supported decoder/summarizer backend. Uses ONNX Runtime "
                        + "GenAI for text generation. INT4 quantized, ~2.3 GB disk."));

        entries.add(new Entry(
                "microsoft/Phi-3.5-mini-instruct-onnx",
                Architecture.CAUSAL_LM,
                "Microsoft",
                "3.8B",
                ChatTemplate.PHI3,
                Status.PLANNED,
                Arrays.asList(
                        "model/phi-3.5-mini-instruct-onnx",
                        "model/microsoft/Phi-3.5-mini-instruct-onnx"),
                "Successor to Phi-3 Mini with improved instruction following. "
                        + "Same architecture; expected to work with ONNX GenAI path."));

        // --- Qwen2.5-Coder: planned until runtime loads/generates ---
        entries.add(new Entry(
                "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                Architecture.CAUSAL_LM,
                "Alibaba/Qwen",
                "0.5B",
                ChatTemplate.CHATML,
                Status.PLANNED,
                Arrays.asList(
                        "model/qwen2.5-coder-0.5b-instruct",
                        "model/Qwen/Qwen2.5-Coder-0.5B-Instruct"),
                "Smallest Qwen2.5-Coder variant. Target for initial Qwen "
                        + "runtime bring-up (code explanation, math reasoning)."));

        entries.add(new Entry(
                "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                Architecture.CAUSAL_LM,
                "Alibaba/Qwen",
                "1.5B",
                ChatTemplate.CHATML,
                Status.PLANNED,
                Arrays.asList(
                        "model/qwen2.5-coder-1.5b-instruct",
                        "model/Qwen/Qwen2.5-Coder-1.5B-Instruct"),
                "Mid-size Qwen2.5-Coder. Scale-up candidate once 0.5B works."));

        entries.add(new Entry(
                "Qwen/Qwen2.5-Coder-3B-Instruct",
                Architecture.CAUSAL_LM,
                "Alibaba/Qwen",
                "3B",
                ChatTemplate.CHATML,
                Status.PLANNED,
                Arrays.asList(
                        "model/qwen2.5-coder-3b-instruct",
                        "model/Qwen/Qwen2.5-Coder-3B-Instruct"),
                "Largest planned Qwen2.5-Coder variant for local deployment."));

        ENTRIES = Collections.unmodifiableList(entries);

        Map<String, Entry> byKey = new LinkedHashMap<String, Entry>();
        for (Entry e : ENTRIES) {
            byKey.put(e.modelId().toLowerCase(Locale.ROOT), e);
        }
        BY_KEY = Collections.unmodifiableMap(byKey);
    }

    private GenerationModelRegistry() {
        // utility class
    }

    /** All known generation model entries in declaration order. */
    public static List<Entry> entries() {
        return ENTRIES;
    }

    /**
     * Subset of {@link #entries()} restricted to a single
     * {@link Architecture}, in declaration order.
     */
    public static List<Entry> entriesByArchitecture(Architecture architecture) {
        if (architecture == null) {
            throw new IllegalArgumentException("architecture must not be null");
        }
        List<Entry> out = new ArrayList<Entry>();
        for (Entry e : ENTRIES) {
            if (e.architecture() == architecture) {
                out.add(e);
            }
        }
        return Collections.unmodifiableList(out);
    }

    /**
     * Subset of {@link #entries()} restricted to a single
     * {@link Status}, in declaration order.
     */
    public static List<Entry> entriesByStatus(Status status) {
        if (status == null) {
            throw new IllegalArgumentException("status must not be null");
        }
        List<Entry> out = new ArrayList<Entry>();
        for (Entry e : ENTRIES) {
            if (e.status() == status) {
                out.add(e);
            }
        }
        return Collections.unmodifiableList(out);
    }

    /**
     * Returns only entries that have an active runtime (shipped or experimental).
     */
    public static List<Entry> runnableEntries() {
        List<Entry> out = new ArrayList<Entry>();
        for (Entry e : ENTRIES) {
            if (e.isRunnable()) {
                out.add(e);
            }
        }
        return Collections.unmodifiableList(out);
    }

    /**
     * Look up an entry by its full model ID (case-insensitive).
     * Returns {@code null} for unknown IDs.
     */
    public static Entry findByModelId(String modelId) {
        if (modelId == null) return null;
        String key = modelId.trim().toLowerCase(Locale.ROOT);
        if (key.isEmpty()) return null;
        return BY_KEY.get(key);
    }
}
