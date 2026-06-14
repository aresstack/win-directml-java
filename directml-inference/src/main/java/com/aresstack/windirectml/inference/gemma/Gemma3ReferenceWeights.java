package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorEntry;
import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorsFile;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * CPU-resident Gemma 3 weights as plain {@code float[]} arrays (parity reference). Loads from the
 * SafeTensors source, decoding F32/F16/BF16 to float. The LM head is tied to {@code embed_tokens}.
 *
 * <p>Heap note: {@code embed_tokens} is {@code vocab*hidden} (≈671 MB FP32 for the real 270M). This
 * reference path is for parity/diagnosis, not the heap-light product path (that is the WARP runtime).</p>
 */
public final class Gemma3ReferenceWeights {

    /** Per-layer weights (row-major HF layout: linear weight is [out, in]). */
    public static final class Layer {
        public final float[] inputLayerNorm;
        public final float[] qProj;
        public final float[] kProj;
        public final float[] vProj;
        public final float[] oProj;
        public final float[] qNorm;
        public final float[] kNorm;
        public final float[] postAttentionLayerNorm;
        public final float[] preFeedforwardLayerNorm;
        public final float[] gateProj;
        public final float[] upProj;
        public final float[] downProj;
        public final float[] postFeedforwardLayerNorm;

        public Layer(float[] inputLayerNorm, float[] qProj, float[] kProj, float[] vProj, float[] oProj,
                     float[] qNorm, float[] kNorm, float[] postAttentionLayerNorm, float[] preFeedforwardLayerNorm,
                     float[] gateProj, float[] upProj, float[] downProj, float[] postFeedforwardLayerNorm) {
            this.inputLayerNorm = inputLayerNorm;
            this.qProj = qProj;
            this.kProj = kProj;
            this.vProj = vProj;
            this.oProj = oProj;
            this.qNorm = qNorm;
            this.kNorm = kNorm;
            this.postAttentionLayerNorm = postAttentionLayerNorm;
            this.preFeedforwardLayerNorm = preFeedforwardLayerNorm;
            this.gateProj = gateProj;
            this.upProj = upProj;
            this.downProj = downProj;
            this.postFeedforwardLayerNorm = postFeedforwardLayerNorm;
        }
    }

    private final Gemma3Config config;
    public final float[] embedTokens;
    public final float[] finalNorm;
    public final Layer[] layers;

    public Gemma3ReferenceWeights(Gemma3Config config, float[] embedTokens, float[] finalNorm, Layer[] layers) {
        this.config = Objects.requireNonNull(config, "config");
        this.embedTokens = Objects.requireNonNull(embedTokens, "embedTokens");
        this.finalNorm = Objects.requireNonNull(finalNorm, "finalNorm");
        this.layers = Objects.requireNonNull(layers, "layers");
    }

    public Gemma3Config config() {
        return config;
    }

    /** {@code embed_tokens} doubles as the (tied) LM head matrix [vocab, hidden]. */
    public float[] lmHead() {
        return embedTokens;
    }

    public static Gemma3ReferenceWeights load(SafeTensorsFile file, Gemma3Config config) throws IOException {
        float[] embed = floats(file, Gemma3TensorNameMapper.EMBED_TOKENS);
        float[] finalNorm = floats(file, Gemma3TensorNameMapper.FINAL_NORM);
        Layer[] layers = new Layer[config.numHiddenLayers()];
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Layer(
                    floats(file, Gemma3TensorNameMapper.inputLayerNorm(i)),
                    floats(file, Gemma3TensorNameMapper.qProj(i)),
                    floats(file, Gemma3TensorNameMapper.kProj(i)),
                    floats(file, Gemma3TensorNameMapper.vProj(i)),
                    floats(file, Gemma3TensorNameMapper.oProj(i)),
                    floats(file, Gemma3TensorNameMapper.qNorm(i)),
                    floats(file, Gemma3TensorNameMapper.kNorm(i)),
                    floats(file, Gemma3TensorNameMapper.postAttentionLayerNorm(i)),
                    floats(file, Gemma3TensorNameMapper.preFeedforwardLayerNorm(i)),
                    floats(file, Gemma3TensorNameMapper.gateProj(i)),
                    floats(file, Gemma3TensorNameMapper.upProj(i)),
                    floats(file, Gemma3TensorNameMapper.downProj(i)),
                    floats(file, Gemma3TensorNameMapper.postFeedforwardLayerNorm(i)));
        }
        return new Gemma3ReferenceWeights(config, embed, finalNorm, layers);
    }

    private static float[] floats(SafeTensorsFile file, String name) throws IOException {
        SafeTensorEntry entry = file.tensors().get(name);
        if (entry == null) {
            throw new IOException("Gemma SafeTensors is missing required tensor: " + name);
        }
        return decodeFloats(entry);
    }

    /** Decode a SafeTensors tensor (F32/F16/BF16) into a float[] (little-endian). */
    public static float[] decodeFloats(SafeTensorEntry entry) throws IOException {
        long count = 1;
        for (long d : entry.shape()) {
            count = Math.multiplyExact(count, d);
        }
        int n = Math.toIntExact(count);
        float[] out = new float[n];
        ByteBuffer b = entry.dataBuffer();
        switch (entry.dtype()) {
            case "F32" -> {
                for (int i = 0; i < n; i++) {
                    out[i] = b.getFloat();
                }
            }
            case "F16" -> {
                for (int i = 0; i < n; i++) {
                    out[i] = Float.float16ToFloat(b.getShort());
                }
            }
            case "BF16" -> {
                for (int i = 0; i < n; i++) {
                    int bits = (b.getShort() & 0xFFFF) << 16; // bf16 = top 16 bits of f32
                    out[i] = Float.intBitsToFloat(bits);
                }
            }
            default -> throw new IOException("Unsupported Gemma weight dtype (expected F32/F16/BF16): " + entry.dtype());
        }
        return out;
    }
}
