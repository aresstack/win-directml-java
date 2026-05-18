package com.aresstack.windirectml.encoder.minilm;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.EncoderArchitecture;
import com.aresstack.windirectml.encoder.EncoderWeights;
import com.aresstack.windirectml.encoder.safetensors.SafetensorsEntry;
import com.aresstack.windirectml.encoder.safetensors.SafetensorsException;
import com.aresstack.windirectml.encoder.safetensors.SafetensorsReader;
import com.aresstack.windirectml.runtime.TensorDataType;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * In-Memory-Repräsentation der MiniLM-Gewichte für den CPU-Forward-Pass.
 * <p>
 * Hält alle benötigten Tensoren als entpackte {@code float[]}-Arrays.
 * BERT-Style-Namen werden gegen mehrere bekannte Präfixe aufgelöst, damit
 * sowohl die rohe HuggingFace-Variante als auch das sentence-transformers-
 * Layout funktionieren.
 */
public final class CpuMiniLmWeights implements EncoderWeights {

    /**
     * Reihenfolge der Präfix-Kandidaten beim Auflösen eines Tensor-Namens.
     */
    private static final String[] PREFIX_CANDIDATES = {
            "",
            "0.auto_model.",
            "bert.",
            "model.",
            "0.auto_model.bert."
    };

    public static final class LayerWeights {
        public final float[] qWeight, qBias;
        public final float[] kWeight, kBias;
        public final float[] vWeight, vBias;
        public final float[] attnOutWeight, attnOutBias;
        public final float[] attnLnGamma, attnLnBeta;
        public final float[] mlpInterWeight, mlpInterBias;
        public final float[] mlpOutWeight, mlpOutBias;
        public final float[] outLnGamma, outLnBeta;

        LayerWeights(float[] qw, float[] qb, float[] kw, float[] kb,
                     float[] vw, float[] vb,
                     float[] ow, float[] ob, float[] ag, float[] ab,
                     float[] iw, float[] ib, float[] mw, float[] mb,
                     float[] og, float[] ob2) {
            this.qWeight = qw;
            this.qBias = qb;
            this.kWeight = kw;
            this.kBias = kb;
            this.vWeight = vw;
            this.vBias = vb;
            this.attnOutWeight = ow;
            this.attnOutBias = ob;
            this.attnLnGamma = ag;
            this.attnLnBeta = ab;
            this.mlpInterWeight = iw;
            this.mlpInterBias = ib;
            this.mlpOutWeight = mw;
            this.mlpOutBias = mb;
            this.outLnGamma = og;
            this.outLnBeta = ob2;
        }
    }

    private final MiniLmArchitecture architecture;
    private final MiniLmConfig config;

    public final float[] wordEmbeddings;        // [vocab, H]
    public final float[] positionEmbeddings;    // [maxPos, H]
    public final float[] tokenTypeEmbeddings;   // [2, H]
    public final float[] embLnGamma;            // [H]
    public final float[] embLnBeta;             // [H]
    public final List<LayerWeights> layers;     // numLayers

    private CpuMiniLmWeights(MiniLmArchitecture arch,
                             float[] we, float[] pe, float[] tte,
                             float[] elng, float[] elnb,
                             List<LayerWeights> layers) {
        this.architecture = arch;
        this.config = arch.config();
        this.wordEmbeddings = we;
        this.positionEmbeddings = pe;
        this.tokenTypeEmbeddings = tte;
        this.embLnGamma = elng;
        this.embLnBeta = elnb;
        this.layers = layers;
    }

    /**
     * Test-only factory. Allows constructing an encoder with synthetic weights.
     */
    public static CpuMiniLmWeights forTesting(MiniLmArchitecture arch,
                                              float[] wordEmb, float[] posEmb, float[] ttEmb,
                                              float[] embLnGamma, float[] embLnBeta,
                                              List<LayerWeights> layers) {
        return new CpuMiniLmWeights(arch, wordEmb, posEmb, ttEmb, embLnGamma, embLnBeta, layers);
    }

    /**
     * Test-only builder for {@link LayerWeights}.
     */
    public static LayerWeights layerForTesting(float[] qw, float[] qb, float[] kw, float[] kb,
                                               float[] vw, float[] vb,
                                               float[] ow, float[] ob, float[] ag, float[] ab,
                                               float[] iw, float[] ib, float[] mw, float[] mb,
                                               float[] og, float[] ob2) {
        return new LayerWeights(qw, qb, kw, kb, vw, vb, ow, ob, ag, ab, iw, ib, mw, mb, og, ob2);
    }

    public static CpuMiniLmWeights load(Path modelDir, MiniLmArchitecture arch) throws EmbeddingException {
        Path safetensors = modelDir.resolve("model.safetensors");
        try (SafetensorsReader reader = SafetensorsReader.open(safetensors)) {
            return loadFromReader(reader, arch);
        } catch (SafetensorsException e) {
            throw new EmbeddingException("Failed to load MiniLM weights from " + safetensors, e);
        }
    }

    static CpuMiniLmWeights loadFromReader(SafetensorsReader reader, MiniLmArchitecture arch)
            throws SafetensorsException, EmbeddingException {
        MiniLmConfig cfg = arch.config();

        String prefix = detectPrefix(reader, "embeddings.word_embeddings.weight");

        float[] we = readFloats(reader, prefix + "embeddings.word_embeddings.weight");
        float[] pe = readFloats(reader, prefix + "embeddings.position_embeddings.weight");
        float[] tte = readFloats(reader, prefix + "embeddings.token_type_embeddings.weight");
        float[] elng = readFloats(reader, prefix + "embeddings.LayerNorm.weight");
        float[] elnb = readFloats(reader, prefix + "embeddings.LayerNorm.bias");

        validateLength("word_embeddings", we, (long) cfg.vocabSize() * cfg.hiddenSize());
        validateLength("position_embeddings", pe, (long) cfg.maxPositionEmbeddings() * cfg.hiddenSize());

        List<LayerWeights> layers = new ArrayList<>(cfg.numLayers());
        for (int i = 0; i < cfg.numLayers(); i++) {
            String base = prefix + "encoder.layer." + i + ".";
            float[] qw = readFloats(reader, base + "attention.self.query.weight");
            float[] qb = readFloats(reader, base + "attention.self.query.bias");
            float[] kw = readFloats(reader, base + "attention.self.key.weight");
            float[] kb = readFloats(reader, base + "attention.self.key.bias");
            float[] vw = readFloats(reader, base + "attention.self.value.weight");
            float[] vb = readFloats(reader, base + "attention.self.value.bias");
            float[] ow = readFloats(reader, base + "attention.output.dense.weight");
            float[] ob = readFloats(reader, base + "attention.output.dense.bias");
            float[] ag = readFloats(reader, base + "attention.output.LayerNorm.weight");
            float[] ab = readFloats(reader, base + "attention.output.LayerNorm.bias");
            float[] iw = readFloats(reader, base + "intermediate.dense.weight");
            float[] ib = readFloats(reader, base + "intermediate.dense.bias");
            float[] mw = readFloats(reader, base + "output.dense.weight");
            float[] mb = readFloats(reader, base + "output.dense.bias");
            float[] og = readFloats(reader, base + "output.LayerNorm.weight");
            float[] ob2 = readFloats(reader, base + "output.LayerNorm.bias");
            layers.add(new LayerWeights(qw, qb, kw, kb, vw, vb, ow, ob, ag, ab,
                    iw, ib, mw, mb, og, ob2));
        }
        return new CpuMiniLmWeights(arch, we, pe, tte, elng, elnb, layers);
    }

    private static String detectPrefix(SafetensorsReader reader, String suffix) throws EmbeddingException {
        for (String candidate : PREFIX_CANDIDATES) {
            if (reader.tensorNames().contains(candidate + suffix)) return candidate;
        }
        throw new EmbeddingException("Could not locate '" + suffix
                + "' under any known prefix " + java.util.Arrays.toString(PREFIX_CANDIDATES)
                + ". Available tensors (sample): "
                + reader.tensorNames().stream().limit(5).toList());
    }

    private static float[] readFloats(SafetensorsReader reader, String name) throws SafetensorsException {
        SafetensorsEntry entry = reader.entry(name);
        if (entry.dataType() != TensorDataType.FLOAT32) {
            throw new SafetensorsException("expected F32 for '" + name + "' but got " + entry.dataType());
        }
        return reader.readFloat32(name);
    }

    private static void validateLength(String name, float[] data, long expected) throws EmbeddingException {
        if (data.length != expected) {
            throw new EmbeddingException("tensor '" + name + "' length " + data.length
                    + " != expected " + expected);
        }
    }

    public MiniLmArchitecture architecture() {
        return architecture;
    }


    /**
     * Convenience to expose the typed config.
     */
    public MiniLmConfig config() {
        return config;
    }

    @Override
    public void close() {
        // float[] sind GC-bar; nichts zu tun
    }
}

