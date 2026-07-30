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
 * {@code tokenizer.json} and {@code reranker.wdmlpack} so wrong model
 * directories fail loudly instead of silently producing bogus scores.
 * <p>
 * The completeness check requires the runtime package
 * ({@code reranker.wdmlpack}), <b>not</b> the raw {@code model.safetensors}:
 * {@link RerankerCpuWeights#load} reads weights exclusively from the
 * package, so a package-only install directory (no raw weights) is a
 * first-class, fully supported layout. Requiring the artefact that is
 * actually consumed keeps the check fail-closed — an un-converted
 * directory that still has only {@code model.safetensors} fails with a
 * clear "run Convert" message instead of passing the check and then
 * failing deep in the loader.
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

    /**
     * Required artefacts for a BERT-WordPiece cross-encoder reranker directory.
     * <p>
     * The weights entry is the runtime package ({@code reranker.wdmlpack}) — the
     * exact artefact {@link RerankerCpuWeights#load} consumes — so a package-only
     * directory (no raw {@code model.safetensors}) loads cleanly, and an
     * un-converted directory fails closed with the repair hint below.
     */
    private static final List<String> REQUIRED_FILES =
            List.of(com.aresstack.windirectml.encoder.pack.EncoderWdmlPack.RERANKER_PACKAGE_FILE,
                    "tokenizer.json", "config.json");

    private static final String REPAIR_HINT =
            "Repair: in the Workbench Download tab, \"Check\" the reranker "
                    + "(ms-marco-MiniLM-L-6-v2) and then \"Convert\" to (re)build "
                    + "reranker.wdmlpack; or force a re-download and convert again.";

    private static void verifyDir(Path modelDir) throws EmbeddingException {
        ModelAssetValidation.requireModelFiles(modelDir, "Reranker", REQUIRED_FILES, REPAIR_HINT);
    }
}

