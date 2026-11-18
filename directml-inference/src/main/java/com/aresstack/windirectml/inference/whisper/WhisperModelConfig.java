package com.aresstack.windirectml.inference.whisper;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Describe the Hugging Face Whisper checkpoint dimensions independently of
 * the inference backend.
 */
public final class WhisperModelConfig {
    private final int melBands;
    private final int modelWidth;
    private final int audioContext;
    private final int textContext;
    private final int encoderLayers;
    private final int decoderLayers;
    private final int encoderHeads;
    private final int decoderHeads;
    private final int encoderFeedForwardWidth;
    private final int decoderFeedForwardWidth;
    private final int vocabularySize;

    private WhisperModelConfig(JsonNode node) throws IOException {
        this.melBands = readInt(node, "num_mel_bins");
        this.modelWidth = readInt(node, "d_model");
        this.audioContext = readInt(node, "max_source_positions");
        this.textContext = readInt(node, "max_target_positions");
        this.encoderLayers = readInt(node, "encoder_layers");
        this.decoderLayers = readInt(node, "decoder_layers");
        this.encoderHeads = readInt(node, "encoder_attention_heads");
        this.decoderHeads = readInt(node, "decoder_attention_heads");
        this.encoderFeedForwardWidth = readInt(node, "encoder_ffn_dim");
        this.decoderFeedForwardWidth = readInt(node, "decoder_ffn_dim");
        this.vocabularySize = readInt(node, "vocab_size");
        if ((melBands != 80 && melBands != 128)
                || modelWidth % encoderHeads != 0
                || modelWidth % decoderHeads != 0
                || audioContext > 1500 || audioContext < 1
                || textContext > 448 || textContext < 1
                || vocabularySize > 100000) {
            throw new IOException("Unsupported or inconsistent Whisper model dimensions");
        }
    }

    public static WhisperModelConfig load(Path configFile) throws IOException {
        JsonNode root = new ObjectMapper().readTree(Objects.requireNonNull(configFile, "configFile").toFile());
        return new WhisperModelConfig(root);
    }

    private static int readInt(JsonNode root, String property) throws IOException {
        JsonNode node = root.get(property);
        if (node == null || !node.canConvertToInt() || node.asInt() <= 0) {
            throw new IOException("Invalid Whisper config integer: " + property);
        }
        return node.asInt();
    }

    public int melBands() { return melBands; }
    public int modelWidth() { return modelWidth; }
    public int audioContext() { return audioContext; }
    public int textContext() { return textContext; }
    public int encoderLayers() { return encoderLayers; }
    public int decoderLayers() { return decoderLayers; }
    public int encoderHeads() { return encoderHeads; }
    public int decoderHeads() { return decoderHeads; }
    public int encoderFeedForwardWidth() { return encoderFeedForwardWidth; }
    public int decoderFeedForwardWidth() { return decoderFeedForwardWidth; }
    public int vocabularySize() { return vocabularySize; }
}
