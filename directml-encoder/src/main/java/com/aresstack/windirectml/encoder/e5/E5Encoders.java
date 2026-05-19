package com.aresstack.windirectml.encoder.e5;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.bert.BertCpuEncoderWeights;
import com.aresstack.windirectml.encoder.bert.BertEncoderConfig;
import com.aresstack.windirectml.encoder.bert.CpuBertEncoder;
import com.aresstack.windirectml.encoder.bert.DirectMlBertEncoder;
import com.aresstack.windirectml.encoder.tokenizer.WordPieceTokenizer;
import com.aresstack.windirectml.runtime.DirectMlContextImpl;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Family-specific convenience loaders for E5 sentence-embedding models.
 * <p>
 * Resolves {@code model.safetensors} and {@code tokenizer.json} from
 * {@code modelDir}, picks the matching {@link BertEncoderConfig}
 * (defaulting to {@link E5EncoderConfig#baseStsEnDe()} for the small
 * en-de variant the project initially targets) and instantiates either
 * the CPU or the DirectML implementation on the shared
 * {@link com.aresstack.windirectml.encoder.bert generic bert pipeline}.
 * <p>
 * Tokenisation re-uses the existing {@link WordPieceTokenizer}; the
 * SentencePiece path needed for {@code multilingual-e5-*} variants is
 * tracked separately and lands as soon as those checkpoints are added
 * to the supported set.
 */
public final class E5Encoders {

    private E5Encoders() {}

    /**
     * Load an E5 model into a {@link CpuBertEncoder}.
     * @param modelDir model directory containing {@code config.json},
     *                 {@code tokenizer.json} and {@code model.safetensors}.
     * @param cfg      target architecture (use {@link E5EncoderConfig}).
     */
    public static CpuBertEncoder loadCpu(Path modelDir, BertEncoderConfig cfg) throws EmbeddingException {
        verifyDir(modelDir);
        try {
            BertCpuEncoderWeights w = BertCpuEncoderWeights.load(modelDir, cfg);
            WordPieceTokenizer t = WordPieceTokenizer.load(modelDir.resolve("tokenizer.json"),
                    cfg.maxPositionEmbeddings());
            return new CpuBertEncoder(cfg, w, t);
        } catch (EmbeddingException e) {
            throw e;
        } catch (Exception e) {
            throw new EmbeddingException("Failed to load CpuBertEncoder(E5) from " + modelDir, e);
        }
    }

    /**
     * Load an E5 model into a {@link DirectMlBertEncoder} with its own
     * {@link DirectMlContextImpl}.
     */
    public static DirectMlBertEncoder loadDirectMl(Path modelDir, BertEncoderConfig cfg)
            throws EmbeddingException {
        verifyDir(modelDir);
        DirectMlContextImpl ctx = null;
        try {
            ctx = new DirectMlContextImpl("directml-e5");
            ctx.initialize();
            if (!ctx.isReady() || !ctx.bindings().hasDirectMl()) {
                throw new EmbeddingException("No DirectML device available on this adapter");
            }
            BertCpuEncoderWeights w = BertCpuEncoderWeights.load(modelDir, cfg);
            WordPieceTokenizer t = WordPieceTokenizer.load(modelDir.resolve("tokenizer.json"),
                    cfg.maxPositionEmbeddings());
            return DirectMlBertEncoder.build(ctx, /* ownsCtx */ true, cfg, w, t);
        } catch (EmbeddingException e) {
            if (ctx != null) try { ctx.close(); } catch (Exception ignored) {}
            throw e;
        } catch (Exception e) {
            if (ctx != null) try { ctx.close(); } catch (Exception ignored) {}
            throw new EmbeddingException("Failed to load DirectMlBertEncoder(E5) from " + modelDir, e);
        }
    }

    /**
     * Loader bound to the {@code danielheinz/e5-base-sts-en-de} preset
     * – the initial E5 target.
     */
    public static DirectMlBertEncoder loadDirectMl(Path modelDir) throws EmbeddingException {
        return loadDirectMl(modelDir, E5EncoderConfig.baseStsEnDe());
    }

    /** Same for the CPU side. */
    public static CpuBertEncoder loadCpu(Path modelDir) throws EmbeddingException {
        return loadCpu(modelDir, E5EncoderConfig.baseStsEnDe());
    }

    private static void verifyDir(Path modelDir) throws EmbeddingException {
        if (modelDir == null || !Files.isDirectory(modelDir)) {
            throw new EmbeddingException("E5 model directory not found: " + modelDir);
        }
        if (!Files.exists(modelDir.resolve("model.safetensors"))) {
            throw new EmbeddingException("E5 model directory missing model.safetensors: " + modelDir);
        }
        if (!Files.exists(modelDir.resolve("tokenizer.json"))) {
            throw new EmbeddingException("E5 model directory missing tokenizer.json: " + modelDir);
        }
    }
}

