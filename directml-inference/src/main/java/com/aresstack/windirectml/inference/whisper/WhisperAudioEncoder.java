package com.aresstack.windirectml.inference.whisper;

import java.io.IOException;

/**
 * Encode Whisper log-Mel features with the pretrained Conv1D and pre-layer-norm
 * Transformer architecture. Read Hugging Face Whisper SafeTensors weights.
 */
public final class WhisperAudioEncoder {
    private final WhisperCheckpoint model;
    private final WhisperModelConfig config;

    public WhisperAudioEncoder(WhisperCheckpoint model, WhisperModelConfig config) {
        this.model = model;
        this.config = config;
    }

    public float[][] encode(float[][] mel) throws IOException {
        if (mel.length != config.melBands() || mel[0].length < 2
                || mel[0].length > config.audioContext() * 2) {
            throw new IllegalArgumentException("Whisper log-Mel input has invalid dimensions");
        }
        float[][] frames = new float[mel[0].length][config.melBands()];
        for (int band = 0; band < mel.length; band++) {
            if (mel[band].length != frames.length) throw new IllegalArgumentException("Mel bands have differing lengths");
            for (int t = 0; t < frames.length; t++) frames[t][band] = mel[band][t];
        }
        float[][] hidden = WhisperMath.convolution(frames,
                WhisperMath.w(model, "model.encoder.conv1"),
                WhisperMath.b(model, "model.encoder.conv1"), 1);
        hidden = WhisperMath.convolution(hidden,
                WhisperMath.w(model, "model.encoder.conv2"),
                WhisperMath.b(model, "model.encoder.conv2"), 2);
        WhisperCheckpoint.Tensor position = model.require("model.encoder.embed_positions.weight");
        position.requireShape(config.audioContext(), config.modelWidth());
        float[] positions = position.values();
        for (int t = 0; t < hidden.length; t++) {
            for (int c = 0; c < config.modelWidth(); c++) {
                hidden[t][c] += positions[t * config.modelWidth() + c];
            }
        }
        for (int layer = 0; layer < config.encoderLayers(); layer++) {
            hidden = encodeBlock(hidden, layer);
        }
        return WhisperMath.norm(hidden,
                WhisperMath.w(model, "model.encoder.layer_norm"),
                WhisperMath.b(model, "model.encoder.layer_norm"));
    }

    private float[][] encodeBlock(float[][] hidden, int layer) throws IOException {
        String base = "model.encoder.layers." + layer + ".";
        float[][] normalized = WhisperMath.norm(hidden,
                WhisperMath.w(model, base + "self_attn_layer_norm"),
                WhisperMath.b(model, base + "self_attn_layer_norm"));
        String attention = base + "self_attn.";
        float[][] queries = WhisperMath.project(normalized,
                WhisperMath.w(model, attention + "q_proj"),
                WhisperMath.b(model, attention + "q_proj"));
        float[][] keys = WhisperMath.project(normalized,
                WhisperMath.w(model, attention + "k_proj"), null);
        float[][] values = WhisperMath.project(normalized,
                WhisperMath.w(model, attention + "v_proj"),
                WhisperMath.b(model, attention + "v_proj"));
        float[][] attended = WhisperMath.attention(queries, keys, values, config.encoderHeads());
        float[][] projected = WhisperMath.project(attended,
                WhisperMath.w(model, attention + "out_proj"),
                WhisperMath.b(model, attention + "out_proj"));
        float[][] residual = WhisperMath.add(hidden, projected);
        float[][] feedInput = WhisperMath.norm(residual,
                WhisperMath.w(model, base + "final_layer_norm"),
                WhisperMath.b(model, base + "final_layer_norm"));
        float[][] feedForward = WhisperMath.feedForward(feedInput,
                WhisperMath.w(model, base + "fc1"), WhisperMath.b(model, base + "fc1"),
                WhisperMath.w(model, base + "fc2"), WhisperMath.b(model, base + "fc2"));
        return WhisperMath.add(residual, feedForward);
    }
}
