package com.aresstack.windirectml.encoder.e5;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.PoolingStrategy;
import com.aresstack.windirectml.encoder.bert.BertConfigJson;
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
 * Each loader takes either an {@link E5Variant} (preset config) or a
 * raw {@link BertEncoderConfig}, locates the required files in
 * {@code modelDir} ({@code config.json}, {@code tokenizer.json},
 * {@code model.safetensors}) and instantiates the corresponding
 * implementation on the shared
 * {@link com.aresstack.windirectml.encoder.bert generic bert pipeline}.
 * <p>
 * If a {@code config.json} is present, the file is parsed and compared
 * against the requested variant config – a mismatch on the shape-relevant
 * axes ({@code hidden_size}, {@code num_hidden_layers},
 * {@code num_attention_heads}, {@code intermediate_size},
 * {@code vocab_size}, {@code type_vocab_size}) is surfaced as a hard
 * {@link EmbeddingException} so the operator gets a clear "wrong
 * variant for this directory" message instead of garbage embeddings.
 * <p>
 * Tokenisation re-uses the existing {@link WordPieceTokenizer}; the
 * SentencePiece path needed for {@code multilingual-e5-*} variants is
 * tracked separately and lands as soon as those checkpoints are added
 * to the supported set.
 */
public final class E5Encoders {

    private E5Encoders() {}

    /**
     * CPU loader for a specific {@link E5Variant}.
     */
    public static CpuBertEncoder loadCpu(Path modelDir, E5Variant variant) throws EmbeddingException {
        return loadCpu(modelDir, resolveConfig(modelDir, variant));
    }

    /**
     * DirectML loader for a specific {@link E5Variant}.
     */
    public static DirectMlBertEncoder loadDirectMl(Path modelDir, E5Variant variant)
            throws EmbeddingException {
        return loadDirectMl(modelDir, resolveConfig(modelDir, variant));
    }

    /**
     * CPU loader for a raw {@link BertEncoderConfig}.
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

    /** DirectML loader for a raw {@link BertEncoderConfig}. */
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
     * Default loader: variant {@link E5Variant#BASE_STS_EN_DE}.
     */
    public static DirectMlBertEncoder loadDirectMl(Path modelDir) throws EmbeddingException {
        return loadDirectMl(modelDir, E5Variant.BASE_STS_EN_DE);
    }

    /** CPU equivalent of {@link #loadDirectMl(Path)}. */
    public static CpuBertEncoder loadCpu(Path modelDir) throws EmbeddingException {
        return loadCpu(modelDir, E5Variant.BASE_STS_EN_DE);
    }

    /**
     * Resolve the effective {@link BertEncoderConfig} for {@code modelDir}
     * given a requested {@link E5Variant}.
     * <p>
     * The on-disk {@code config.json} is <b>required</b> for E5 model
     * directories: the soft fields ({@code modelName},
     * {@code maxPositionEmbeddings}) come from the file, and the
     * shape-relevant axes are then verified against the variant's
     * declared dimensions. If the directory has no {@code config.json},
     * a hard {@link EmbeddingException} is thrown – we refuse to load
     * E5 weights "blindly" against a hard-coded variant config because
     * that is exactly how silent path/config mismatches sneak in.
     */
    public static BertEncoderConfig resolveConfig(Path modelDir, E5Variant variant)
            throws EmbeddingException {
        BertEncoderConfig declared = variant.config();
        Path configJson = modelDir.resolve("config.json");
        if (!Files.exists(configJson)) {
            throw new EmbeddingException(
                    "E5 model directory is missing config.json: " + modelDir
                            + " (E5 requires config.json so path and -De5.model="
                            + variant.token() + " can be verified against each other; "
                            + "re-run scripts/download-e5.ps1 -Variant " + variant.token() + ")");
        }
        BertEncoderConfig onDisk = BertConfigJson.read(modelDir, declared.modelName(),
                PoolingStrategy.MEAN, /* normalize */ true);
        BertConfigJson.verifyMatches(declared, onDisk, modelDir);
        return onDisk;
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



