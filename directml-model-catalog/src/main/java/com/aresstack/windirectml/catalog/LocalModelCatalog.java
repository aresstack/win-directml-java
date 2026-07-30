package com.aresstack.windirectml.catalog;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The authoritative neutral catalog of win-directml model families that a local host (AskAI, the workbench)
 * may install and run. Each entry carries its full {@link LocalRuntimeModelDescriptor} — capabilities,
 * source format, download manifest, runtime family, wdmlpack package name and supported backends — so the
 * host never re-declares model metadata or infers a capability from a name.
 *
 * <p>Only {@link ModelStatus#RUNNABLE} entries are offered as a local recommendation. A checkpoint whose
 * runtime path exists but has not passed a real compile + load + task smoke stays {@link ModelStatus#UNVERIFIED}
 * (e.g. the L-12 reranker) and is deliberately NOT offered until it is proven.</p>
 *
 * <p>Java-8 compatible.</p>
 */
public final class LocalModelCatalog {

    private static final List<LocalRuntimeModelDescriptor> ENTRIES;
    private static final Map<String, LocalRuntimeModelDescriptor> BY_REPO;

    static {
        List<LocalRuntimeModelDescriptor> e = new ArrayList<LocalRuntimeModelDescriptor>();

        // ---- Embeddings (BERT/WordPiece encoders → encoder.wdmlpack; CPU reference + DirectML) ----
        e.add(encoder("sentence-transformers/all-MiniLM-L6-v2", CatalogModelFamily.MINILM,
                "MINILM_L6_V2", "all-MiniLM-L6-v2",
                "Default fast embedding model (WordPiece, BERT-style)."));
        e.add(encoder("intfloat/e5-small-v2", CatalogModelFamily.E5,
                "E5_SMALL_V2", "e5-small-v2",
                "E5 small (WordPiece, query:/passage: prefixes)."));
        e.add(encoder("intfloat/e5-base-v2", CatalogModelFamily.E5,
                "E5_BASE_V2", "e5-base-v2",
                "E5 base (WordPiece, query:/passage: prefixes)."));
        e.add(encoder("intfloat/e5-large-v2", CatalogModelFamily.E5,
                "E5_LARGE_V2", "e5-large-v2",
                "E5 large (WordPiece); heavier, CPU + DirectML."));

        // ---- Reranker (BERT cross-encoder + linear head → reranker.wdmlpack) ----
        e.add(reranker("cross-encoder/ms-marco-MiniLM-L6-v2", "MS_MARCO_MINILM_L6",
                "cross-encoder-ms-marco-MiniLM-L-6-v2", ModelStatus.RUNNABLE,
                "Shipped ms-marco cross-encoder (RAW_LOGIT scores)."));
        // L-12 stays UNVERIFIED (NOT a local recommendation) until a real compile + CPU + DirectML/WARP +
        // ranking smoke proves it; only then may it be promoted to RUNNABLE. The former package-only load
        // blocker (P3 — the loader demanded model.safetensors) is fixed: package-only loading now works on
        // CPU (proven with the L6 package) and the L12 package-only investigation (RerankerL12CertificationIT)
        // asserts the success path. Promotion is gated only on running that opt-in real CPU + WARP smoke green.
        e.add(reranker("cross-encoder/ms-marco-MiniLM-L12-v2", "MS_MARCO_MINILM_L12",
                "cross-encoder-ms-marco-MiniLM-L-12-v2", ModelStatus.UNVERIFIED,
                "Needs the real CPU + DirectML/WARP package-only ranking smoke to run green before it may "
                        + "become a RUNNABLE local recommendation (package-only load blocker P3 is fixed)."));

        // ---- Causal LM ----
        // Qwen2.5-Coder-0.5B: the runnable path is the ONNX INT4 (Q4F16) build from onnx-community, compiled
        // to model_q4f16.wdmlpack. The canonical display repo stays Qwen/…; the weights come from the
        // onnx-community mirror under the `onnx` subdirectory.
        e.add(LocalRuntimeModelDescriptor.builder("Qwen/Qwen2.5-Coder-0.5B-Instruct",
                        CatalogModelFamily.QWEN, ModelStatus.RUNNABLE)
                .runtimeModelId("QWEN2_5_CODER_0_5B_INSTRUCT")
                .architecture("causal-lm")
                .capabilities(ModelCapability.COMPLETION, ModelCapability.CHAT)
                .sourceFormat(SourceFormat.ONNX_INT4)
                .tokenizerFamily("bpe")
                .chatTemplate("chatml")
                // QwenInferenceEngine accepts warp/auto/cpu (+directml/hybrid); the workbench offers WARP
                // (software adapter) and AUTO (hardware adapter), with a real CPU reference decode path.
                .backends(CatalogBackend.WARP, CatalogBackend.AUTO, CatalogBackend.CPU)
                .runtimeDirectoryName("qwen2.5-coder-0.5b-directml-int4")
                .downloadManifest(new DownloadManifest("onnx-community/Qwen2.5-Coder-0.5B-Instruct",
                        Arrays.asList(
                                new DownloadFile("onnx/model_q4f16.onnx", "model_q4f16.onnx", true),
                                DownloadFile.required("config.json"),
                                DownloadFile.required("tokenizer.json"),
                                DownloadFile.required("tokenizer_config.json"),
                                DownloadFile.optional("generation_config.json"),
                                DownloadFile.optional("vocab.json"),
                                DownloadFile.optional("merges.txt"),
                                DownloadFile.optional("special_tokens_map.json"))))
                .notes("Native DirectML INT4 runtime (QwenInferenceEngine) from model_q4f16.wdmlpack; no Python.")
                .build());

        e.add(smolLm2("HuggingFaceTB/SmolLM2-135M-Instruct", "SMOLLM2_135M_INSTRUCT", "smollm2-135m-instruct"));
        e.add(smolLm2("HuggingFaceTB/SmolLM2-360M-Instruct", "SMOLLM2_360M_INSTRUCT", "smollm2-360m-instruct"));

        // Gemma 3 270M-it: native WARP (software) + AUTO (hardware) only. The CPU path in the workbench is an
        // external Python/Transformers bridge and is deliberately EXCLUDED here, so a strict host never
        // silently falls back to Python. Gated repository → requires an authenticated HuggingFace token.
        e.add(LocalRuntimeModelDescriptor.builder("google/gemma-3-270m-it",
                        CatalogModelFamily.GEMMA3, ModelStatus.RUNNABLE)
                .runtimeModelId("GEMMA3_270M_IT")
                .architecture("causal-lm")
                .capabilities(ModelCapability.COMPLETION, ModelCapability.CHAT)
                .sourceFormat(SourceFormat.SAFETENSORS)
                .tokenizerFamily("sentencepiece")
                .chatTemplate("gemma3")
                .backends(CatalogBackend.WARP, CatalogBackend.AUTO)
                .gated(true)
                .downloadManifest(new DownloadManifest("google/gemma-3-270m-it", Arrays.asList(
                        DownloadFile.required("model.safetensors"),
                        DownloadFile.required("config.json"),
                        DownloadFile.required("tokenizer.json"),
                        DownloadFile.required("tokenizer.model"),
                        DownloadFile.required("tokenizer_config.json"),
                        DownloadFile.required("special_tokens_map.json"),
                        DownloadFile.optional("added_tokens.json"),
                        DownloadFile.optional("generation_config.json"),
                        DownloadFile.optional("chat_template.jinja"))))
                .notes("Native Java/DirectML Gemma 3 via model_gemma3.wdmlpack. WARP (software adapter) or AUTO "
                        + "(hardware) only — the CPU path is an external Python bridge and is NOT offered here.")
                .build());

        // Phi-3 mini 4k INT4 ONNX (DirectML), compiled to model_phi3.wdmlpack; heap-light CPU load + DirectML.
        e.add(LocalRuntimeModelDescriptor.builder("microsoft/Phi-3-mini-4k-instruct-onnx",
                        CatalogModelFamily.PHI3, ModelStatus.RUNNABLE)
                .runtimeModelId("PHI3_MINI_4K_INSTRUCT")
                .architecture("causal-lm")
                .capabilities(ModelCapability.COMPLETION, ModelCapability.CHAT)
                .sourceFormat(SourceFormat.ONNX_INT4)
                .tokenizerFamily("sentencepiece")
                .chatTemplate("phi3")
                // The neutral runtime loads Phi-3 package-backed (model_phi3.wdmlpack via Phi3Runtime),
                // which currently has only a CPU compute path — the DirectML kernels in
                // Phi3InferenceEngine are the raw-ONNX (non-package) path and are not wired to the
                // package weights yet (see problems.md). The matrix is limited to the truly working
                // package-backed backend; DIRECTML/AUTO return to the matrix once wired + real-tested.
                .backends(CatalogBackend.CPU)
                .runtimeDirectoryName("phi-3-mini-4k-instruct-onnx")
                .downloadManifest(new DownloadManifest("microsoft/Phi-3-mini-4k-instruct-onnx", Arrays.asList(
                        new DownloadFile("directml/directml-int4-awq-block-128/model.onnx", "model.onnx", true),
                        new DownloadFile("directml/directml-int4-awq-block-128/model.onnx.data", "model.onnx.data", true),
                        new DownloadFile("directml/directml-int4-awq-block-128/tokenizer.json", "tokenizer.json", true),
                        new DownloadFile("directml/directml-int4-awq-block-128/config.json", "config.json", true))))
                .notes("Native Java/DirectML Phi-3 decoder from model_phi3.wdmlpack; no Python/ONNX Runtime.")
                .build());

        // ---- Seq2Seq (T5 family → model_t5.wdmlpack) ----
        e.add(t5SafeTensors("google-t5/t5-small", "T5_SMALL", "t5-small"));
        e.add(t5SafeTensors("google/flan-t5-small", "FLAN_T5_SMALL", "flan-t5-small"));
        e.add(codeT5Torch("Salesforce/codet5-small", "CODET5_SMALL", "codet5-small"));
        e.add(codeT5Torch("Salesforce/codet5-base-multi-sum", "CODET5_BASE_MULTI_SUM", "codet5-base-multi-sum"));

        ENTRIES = Collections.unmodifiableList(e);
        Map<String, LocalRuntimeModelDescriptor> byRepo = new LinkedHashMap<String, LocalRuntimeModelDescriptor>();
        for (LocalRuntimeModelDescriptor d : ENTRIES) {
            byRepo.put(d.huggingFaceRepositoryId().toLowerCase(Locale.ROOT), d);
        }
        BY_REPO = Collections.unmodifiableMap(byRepo);
    }

    private LocalModelCatalog() {
    }

    // ---- family builders ----

    private static LocalRuntimeModelDescriptor encoder(String repo, CatalogModelFamily family,
                                                       String runtimeModelId, String dirName, String notes) {
        return LocalRuntimeModelDescriptor.builder(repo, family, ModelStatus.RUNNABLE)
                .runtimeModelId(runtimeModelId)
                .architecture("bert-encoder")
                .capabilities(ModelCapability.EMBEDDING)
                .sourceFormat(SourceFormat.SAFETENSORS)
                .tokenizerFamily("wordpiece")
                .backends(CatalogBackend.CPU, CatalogBackend.DIRECTML)
                .runtimeDirectoryName(dirName)
                .downloadManifest(new DownloadManifest(repo, LocalRuntimeModelDescriptor.filesAtRoot(
                        new String[]{"model.safetensors", "tokenizer.json", "config.json"},
                        new String[]{"vocab.txt", "special_tokens_map.json", "tokenizer_config.json"})))
                .notes(notes)
                .build();
    }

    private static LocalRuntimeModelDescriptor reranker(String repo, String runtimeModelId, String dirName,
                                                        ModelStatus status, String notes) {
        LocalRuntimeModelDescriptor.Builder b = LocalRuntimeModelDescriptor.builder(repo,
                        CatalogModelFamily.CROSS_ENCODER, status)
                .runtimeModelId(runtimeModelId)
                .architecture("bert-cross-encoder")
                .capabilities(ModelCapability.RERANK)
                .sourceFormat(SourceFormat.SAFETENSORS)
                .tokenizerFamily("wordpiece")
                .backends(CatalogBackend.CPU, CatalogBackend.DIRECTML)
                .runtimeDirectoryName(dirName)
                .notes(notes);
        if (status == ModelStatus.RUNNABLE) {
            b.downloadManifest(new DownloadManifest(repo, LocalRuntimeModelDescriptor.filesAtRoot(
                    new String[]{"model.safetensors", "tokenizer.json", "config.json"},
                    new String[]{"vocab.txt", "special_tokens_map.json", "tokenizer_config.json"})));
        }
        return b.build();
    }

    private static LocalRuntimeModelDescriptor smolLm2(String repo, String runtimeModelId, String dirName) {
        return LocalRuntimeModelDescriptor.builder(repo, CatalogModelFamily.SMOLLM2, ModelStatus.RUNNABLE)
                .runtimeModelId(runtimeModelId)
                .architecture("causal-lm")
                .capabilities(ModelCapability.COMPLETION, ModelCapability.CHAT)
                .sourceFormat(SourceFormat.SAFETENSORS)
                .tokenizerFamily("bpe")
                .chatTemplate("raw")
                // Real-model certification: CPU (reference) and AUTO (native DirectML on a hardware
                // adapter, else reference) are green. The D3D12 WARP *software* rasterizer produced
                // empty output with the real weights (see problems.md / LOCAL_ENGINE_CERTIFICATION.md),
                // so WARP is withheld from the matrix rather than offered while broken.
                .backends(CatalogBackend.AUTO, CatalogBackend.CPU)
                .runtimeDirectoryName(dirName)
                .downloadManifest(new DownloadManifest(repo, LocalRuntimeModelDescriptor.filesAtRoot(
                        new String[]{"model.safetensors", "tokenizer.json", "config.json",
                                "tokenizer_config.json", "special_tokens_map.json"},
                        new String[]{"generation_config.json", "merges.txt", "vocab.json"})))
                .notes("Native DirectML/WARP SmolLM2 (dense projections on the D3D12 software rasterizer; "
                        + "CPU reference fallback); no Python.")
                .build();
    }

    private static LocalRuntimeModelDescriptor t5SafeTensors(String repo, String runtimeModelId, String dirName) {
        return LocalRuntimeModelDescriptor.builder(repo, CatalogModelFamily.T5, ModelStatus.RUNNABLE)
                .runtimeModelId(runtimeModelId)
                .architecture("seq2seq")
                .capabilities(ModelCapability.SEQ2SEQ, ModelCapability.COMPLETION, ModelCapability.SUMMARIZE)
                .sourceFormat(SourceFormat.SAFETENSORS)
                .tokenizerFamily("sentencepiece")
                .chatTemplate("raw")
                .backends(CatalogBackend.WARP, CatalogBackend.AUTO, CatalogBackend.CPU)
                .runtimeDirectoryName(dirName)
                .downloadManifest(new DownloadManifest(repo, LocalRuntimeModelDescriptor.filesAtRoot(
                        new String[]{"model.safetensors", "config.json", "tokenizer.json",
                                "tokenizer_config.json", "spiece.model"},
                        new String[]{"special_tokens_map.json", "generation_config.json"})))
                .notes("Encoder-decoder T5 via model_t5.wdmlpack (SafeTensors import).")
                .build();
    }

    private static LocalRuntimeModelDescriptor codeT5Torch(String repo, String runtimeModelId, String dirName) {
        return LocalRuntimeModelDescriptor.builder(repo, CatalogModelFamily.T5, ModelStatus.RUNNABLE)
                .runtimeModelId(runtimeModelId)
                .architecture("seq2seq")
                .capabilities(ModelCapability.SEQ2SEQ, ModelCapability.COMPLETION, ModelCapability.SUMMARIZE)
                .sourceFormat(SourceFormat.TORCH_CHECKPOINT)
                .tokenizerFamily("bpe")
                .chatTemplate("raw")
                .backends(CatalogBackend.WARP, CatalogBackend.AUTO, CatalogBackend.CPU)
                .runtimeDirectoryName(dirName)
                .downloadManifest(new DownloadManifest(repo, LocalRuntimeModelDescriptor.filesAtRoot(
                        new String[]{"pytorch_model.bin", "config.json", "vocab.json", "merges.txt",
                                "tokenizer_config.json", "special_tokens_map.json"},
                        new String[]{"added_tokens.json"})))
                .notes("CodeT5 via the restricted pytorch_model.bin import path → model_t5.wdmlpack.")
                .build();
    }

    // ---- queries ----

    /** Every catalog entry in declaration order (including UNVERIFIED/non-runnable ones). */
    public static List<LocalRuntimeModelDescriptor> entries() {
        return ENTRIES;
    }

    /** Only entries a host may offer as a local install recommendation and run productively. */
    public static List<LocalRuntimeModelDescriptor> runnable() {
        List<LocalRuntimeModelDescriptor> out = new ArrayList<LocalRuntimeModelDescriptor>();
        for (LocalRuntimeModelDescriptor d : ENTRIES) {
            if (d.isRunnable()) {
                out.add(d);
            }
        }
        return Collections.unmodifiableList(out);
    }

    /** Runnable entries that advertise the given capability, in declaration order. */
    public static List<LocalRuntimeModelDescriptor> runnableByCapability(ModelCapability capability) {
        List<LocalRuntimeModelDescriptor> out = new ArrayList<LocalRuntimeModelDescriptor>();
        for (LocalRuntimeModelDescriptor d : ENTRIES) {
            if (d.isRunnable() && d.hasCapability(capability)) {
                out.add(d);
            }
        }
        return Collections.unmodifiableList(out);
    }

    /** Look up an entry by its display HuggingFace repository id (case-insensitive), or {@code null}. */
    public static LocalRuntimeModelDescriptor findByRepositoryId(String repositoryId) {
        if (repositoryId == null) {
            return null;
        }
        String key = repositoryId.trim().toLowerCase(Locale.ROOT);
        if (key.isEmpty()) {
            return null;
        }
        return BY_REPO.get(key);
    }
}
