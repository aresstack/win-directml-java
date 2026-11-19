package com.aresstack.windirectml.inference.whisper;

import java.io.IOException;

/**
 * Decode Whisper tokens autoregressively with cached self-attention and
 * per-layer, precomputed encoder cross-attention key/value tensors.
 */
public final class WhisperTextDecoder {
    private static final int LOGIT_BLOCK = 2048;

    private final WhisperCheckpoint checkpoint;
    private final WhisperModelConfig config;
    private final WhisperCheckpoint.Tensor embeddings;
    private final WhisperCheckpoint.Tensor positions;
    private final WhisperCheckpoint.Tensor finalNormWeight;
    private final WhisperCheckpoint.Tensor finalNormBias;

    public WhisperTextDecoder(WhisperCheckpoint checkpoint, WhisperModelConfig config) throws IOException {
        this.checkpoint = checkpoint;
        this.config = config;
        this.embeddings = checkpoint.require("model.decoder.embed_tokens.weight");
        this.positions = checkpoint.require("model.decoder.embed_positions.weight");
        this.finalNormWeight = WhisperMath.w(checkpoint, "model.decoder.layer_norm");
        this.finalNormBias = WhisperMath.b(checkpoint, "model.decoder.layer_norm");
        embeddings.requireShape(config.vocabularySize(), config.modelWidth());
        positions.requireShape(config.textContext(), config.modelWidth());
    }

    public Session newSession(float[][] encodedAudio) throws IOException {
        return newSession(prepare(encodedAudio));
    }

    /** Project encoder features to cross-attention keys and values once per audio window. */
    public EncodedAudio prepare(float[][] encodedAudio) throws IOException {
        if (encodedAudio == null || encodedAudio.length == 0 || encodedAudio[0].length != config.modelWidth()) {
            throw new IllegalArgumentException("Whisper encoder features have incompatible dimensions");
        }
        float[][][] keys = new float[config.decoderLayers()][][];
        float[][][] values = new float[config.decoderLayers()][][];
        for (int layer = 0; layer < keys.length; layer++) {
            String crossBase = "model.decoder.layers." + layer + ".encoder_attn.";
            keys[layer] = WhisperMath.project(encodedAudio,
                    WhisperMath.w(checkpoint, crossBase + "k_proj"), null);
            values[layer] = WhisperMath.project(encodedAudio,
                    WhisperMath.w(checkpoint, crossBase + "v_proj"),
                    WhisperMath.b(checkpoint, crossBase + "v_proj"));
        }
        return new EncodedAudio(keys, values);
    }

    /** Start an independent generation that reuses prepared cross-attention tensors. */
    public Session newSession(EncodedAudio audio) throws IOException {
        Layer[] layers = new Layer[config.decoderLayers()];
        for (int layer = 0; layer < layers.length; layer++)
            layers[layer] = new Layer(checkpoint, config, layer,
                    audio.crossKeys[layer], audio.crossValues[layer]);
        return new Session(layers);
    }

    /** Hold the immutable cross-attention projections of one encoded audio window. */
    public static final class EncodedAudio {
        private final float[][][] crossKeys;
        private final float[][][] crossValues;

        private EncodedAudio(float[][][] crossKeys, float[][][] crossValues) {
            this.crossKeys = crossKeys;
            this.crossValues = crossValues;
        }
    }

    /** Keep the mutable generation state scoped to one transcription. */
    public final class Session {
        private final Layer[] layers;
        private int position;

        private Session(Layer[] layers) {
            this.layers = layers;
        }

        public float[] nextLogits(int lastToken) {
            if (position >= config.textContext()) {
                throw new IllegalStateException("Whisper text context exceeded");
            }
            if (lastToken < 0 || lastToken >= config.vocabularySize()) {
                throw new IllegalArgumentException("Invalid Whisper token: " + lastToken);
            }
            int width = config.modelWidth();
            float[] hidden = new float[width];
            float[] tokenWeights = embeddings.values();
            float[] positionWeights = positions.values();
            int tokenOffset = lastToken * width;
            int posOffset = position * width;
            for (int i = 0; i < width; i++) {
                hidden[i] = tokenWeights[tokenOffset + i] + positionWeights[posOffset + i];
            }
            for (Layer layer : layers) hidden = layer.step(hidden, position);
            float[] output = WhisperMath.norm(hidden, finalNormWeight, finalNormBias);
            position++;
            // Whisper ties its LM head to the decoder token-embedding matrix.
            float[] logits = new float[config.vocabularySize()];
            int blocks = (logits.length + LOGIT_BLOCK - 1) / LOGIT_BLOCK;
            WhisperMath.forEachRow(blocks, block -> {
                int end = Math.min(logits.length, (block + 1) * LOGIT_BLOCK);
                for (int token = block * LOGIT_BLOCK; token < end; token++) {
                    int offset = token * width;
                    double sum = 0;
                    for (int i = 0; i < width; i++) sum += output[i] * tokenWeights[offset + i];
                    logits[token] = (float) sum;
                }
            });
            return logits;
        }

        public int tokensConsumed() {
            return position;
        }
    }

    private static final class Layer {
        private final int attentionHeads;
        private final WhisperCheckpoint.Tensor selfNormWeight, selfNormBias;
        private final WhisperCheckpoint.Tensor selfQ, selfQB, selfK, selfV, selfVB, selfOut, selfOutB;
        private final WhisperCheckpoint.Tensor crossNormWeight, crossNormBias;
        private final WhisperCheckpoint.Tensor crossQ, crossQB, crossOut, crossOutB;
        private final WhisperCheckpoint.Tensor finalNormWeight, finalNormBias, fc1, fc1B, fc2, fc2B;
        private final float[][] selfKeys, selfValues, crossKeys, crossValues;

        private Layer(WhisperCheckpoint model, WhisperModelConfig config, int layerIndex,
                      float[][] crossKeys, float[][] crossValues) throws IOException {
            String base = "model.decoder.layers." + layerIndex + ".";
            attentionHeads = config.decoderHeads();
            selfNormWeight = WhisperMath.w(model, base + "self_attn_layer_norm");
            selfNormBias = WhisperMath.b(model, base + "self_attn_layer_norm");
            String selfBase = base + "self_attn.";
            selfQ = WhisperMath.w(model, selfBase + "q_proj");
            selfQB = WhisperMath.b(model, selfBase + "q_proj");
            selfK = WhisperMath.w(model, selfBase + "k_proj");
            selfV = WhisperMath.w(model, selfBase + "v_proj");
            selfVB = WhisperMath.b(model, selfBase + "v_proj");
            selfOut = WhisperMath.w(model, selfBase + "out_proj");
            selfOutB = WhisperMath.b(model, selfBase + "out_proj");
            crossNormWeight = WhisperMath.w(model, base + "encoder_attn_layer_norm");
            crossNormBias = WhisperMath.b(model, base + "encoder_attn_layer_norm");
            String crossBase = base + "encoder_attn.";
            crossQ = WhisperMath.w(model, crossBase + "q_proj");
            crossQB = WhisperMath.b(model, crossBase + "q_proj");
            crossOut = WhisperMath.w(model, crossBase + "out_proj");
            crossOutB = WhisperMath.b(model, crossBase + "out_proj");
            this.crossKeys = crossKeys;
            this.crossValues = crossValues;
            selfKeys = new float[config.textContext()][];
            selfValues = new float[config.textContext()][];
            finalNormWeight = WhisperMath.w(model, base + "final_layer_norm");
            finalNormBias = WhisperMath.b(model, base + "final_layer_norm");
            fc1 = WhisperMath.w(model, base + "fc1");
            fc1B = WhisperMath.b(model, base + "fc1");
            fc2 = WhisperMath.w(model, base + "fc2");
            fc2B = WhisperMath.b(model, base + "fc2");
        }

        private float[] step(float[] hidden, int position) {
            float[] normalizedSelf = WhisperMath.norm(hidden, selfNormWeight, selfNormBias);
            float[] query = WhisperMath.project(normalizedSelf, selfQ, selfQB);
            selfKeys[position] = WhisperMath.project(normalizedSelf, selfK, null);
            selfValues[position] = WhisperMath.project(normalizedSelf, selfV, selfVB);
            float[] context = WhisperMath.attention(query, selfKeys, selfValues,
                    attentionHeads, position + 1);
            float[] selfProjected = WhisperMath.project(context, selfOut, selfOutB);
            float[] afterSelf = WhisperMath.add(hidden, selfProjected);

            float[] normalizedCross = WhisperMath.norm(afterSelf, crossNormWeight, crossNormBias);
            float[] crossQuery = WhisperMath.project(normalizedCross, crossQ, crossQB);
            float[] crossContext = WhisperMath.attention(crossQuery, crossKeys, crossValues,
                    attentionHeads, crossKeys.length);
            float[] afterCross = WhisperMath.add(afterSelf,
                    WhisperMath.project(crossContext, crossOut, crossOutB));

            float[] normalizedFeed = WhisperMath.norm(afterCross, finalNormWeight, finalNormBias);
            return WhisperMath.add(afterCross,
                    WhisperMath.feedForward(normalizedFeed, fc1, fc1B, fc2, fc2B));
        }
    }
}
