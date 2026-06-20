package com.aresstack.windirectml.encoder.reranker;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.ModelAssetValidation;
import com.aresstack.windirectml.encoder.PoolingStrategy;
import com.aresstack.windirectml.encoder.bert.BertConfigJson;
import com.aresstack.windirectml.encoder.bert.BertEncoderConfig;
import com.aresstack.windirectml.encoder.tokenizer.WordPieceTokenizer;
import com.aresstack.windirectml.runtime.DirectMlContextImpl;

import java.nio.file.Path;
import java.util.List;

/**
 * Family-specific convenience loaders for BERT-based cross-encoder
 * rerankers ({@code cross-encoder/ms-marco-MiniLM-L-*-v2},
 * {@code BAAI/bge-reranker-base}, …).
 * <p>
 * Each loader reads {@code modelDir/config.json} via
 * {@link BertConfigJson} – there is no hard-coded variant table yet
 * because the reranker landscape is more heterogeneous than the E5
 * embedding family, and the on-disk config is the single source of
 * truth anyway. We do <b>require</b> a present {@code config.json},
 * {@code tokenizer.json} and {@code model.safetensors} so wrong model
 * directories fail loudly instead of silently producing bogus scores.
 * <p>
 * Tokenisation re-uses the existing {@link WordPieceTokenizer}; once
 * SentencePiece-based rerankers (e.g. bge-reranker-v2-m3) are added
 * we will route them to the dedicated SentencePiece tokenizer instead.
 */
public final class BertCrossEncoderRerankers {

    private BertCrossEncoderRerankers() {
    }

    /**
     * CPU loader.
     */
    public static CpuReranker loadCpu(Path modelDir) throws EmbeddingException {
        verifyDir(modelDir);
        BertEncoderConfig cfg = readConfig(modelDir);
        try {
            RerankerCpuWeights w = RerankerCpuWeights.load(modelDir, cfg);
            WordPieceTokenizer t = WordPieceTokenizer.load(modelDir.resolve("tokenizer.json"),
                    cfg.maxPositionEmbeddings());
            return new CpuReranker(w, t);
        } catch (EmbeddingException e) {
            throw e;
        } catch (Exception e) {
            throw new EmbeddingException("Failed to load CpuReranker from " + modelDir, e);
        }
    }

    /**
     * DirectML loader. Owns the context so a single {@code close()} cleans up everything.
     */
    public static DirectMlReranker loadDirectMl(Path modelDir) throws EmbeddingException {
        return loadDirectMl(modelDir, "directml");
    }

    /**
     * DirectML loader with an explicit native backend.
     */
    public static DirectMlReranker loadDirectMl(Path modelDir, String nativeBackend) throws EmbeddingException {
        verifyDir(modelDir);
        BertEncoderConfig cfg = readConfig(modelDir);
        DirectMlContextImpl ctx = null;
        try {
            ctx = new DirectMlContextImpl(normalizeNativeBackend(nativeBackend));
            ctx.initialize();
            if (!ctx.isReady() || !ctx.bindings().hasDirectMl()) {
                throw new EmbeddingException("No DirectML device available on this adapter");
            }
            RerankerCpuWeights w = RerankerCpuWeights.load(modelDir, cfg);
            WordPieceTokenizer t = WordPieceTokenizer.load(modelDir.resolve("tokenizer.json"),
                    cfg.maxPositionEmbeddings());
            return DirectMlReranker.build(ctx, /* ownsCtx */ true, w, t);
        } catch (EmbeddingException e) {
            if (ctx != null) try {
                ctx.close();
            } catch (Exception ignored) {
            }
            throw e;
        } catch (RerankException e) {
            if (ctx != null) try {
                ctx.close();
            } catch (Exception ignored) {
            }
            throw new EmbeddingException("Failed to build DirectMlReranker from " + modelDir, e);
        } catch (Exception e) {
            if (ctx != null) try {
                ctx.close();
            } catch (Exception ignored) {
            }
            throw new EmbeddingException("Failed to load DirectMlReranker from " + modelDir, e);
        }
    }

    private static String normalizeNativeBackend(String nativeBackend) {
        if (nativeBackend == null || nativeBackend.trim().isEmpty()) {
            return "directml";
        }
        return nativeBackend.trim();
    }

    private static BertEncoderConfig readConfig(Path modelDir) throws EmbeddingException {
        // PoolingStrategy.MEAN keeps BertEncoderConfig.validate() happy –
        // it is structurally unused by the reranker path (cross-encoders
        // always use the [CLS] hidden state).
        return BertConfigJson.read(modelDir, modelDir.getFileName().toString(),
                PoolingStrategy.MEAN, /* normalize */ false);
    }

    /** Required artefacts for a BERT-WordPiece cross-encoder reranker directory. */
    private static final List<String> REQUIRED_FILES =
            List.of("model.safetensors", "tokenizer.json", "config.json");

    private static final String REPAIR_HINT =
            "Repair: re-download via the Workbench Download tab (\"Download Reranker "
                    + "(ms-marco-MiniLM-L-6-v2)\") with \"Force re-download\" enabled, "
                    + "or delete the folder and download again.";

    private static void verifyDir(Path modelDir) throws EmbeddingException {
        ModelAssetValidation.requireModelFiles(modelDir, "Reranker", REQUIRED_FILES, REPAIR_HINT);
    }
}

