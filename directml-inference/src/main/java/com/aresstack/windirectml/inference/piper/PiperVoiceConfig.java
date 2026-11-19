package com.aresstack.windirectml.inference.piper;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/** Read and validate the Piper voice contract independently of the ONNX execution backend. */
public final class PiperVoiceConfig {

    private final int sampleRate;
    private final int speakerCount;
    private final String espeakVoice;
    private final Map<String, int[]> phonemeIds;
    private final float noiseScale;
    private final float lengthScale;
    private final float noiseScaleW;

    private PiperVoiceConfig(int sampleRate, int speakerCount, String espeakVoice,
                             Map<String, int[]> phonemeIds, float noiseScale,
                             float lengthScale, float noiseScaleW) {
        this.sampleRate = sampleRate;
        this.speakerCount = speakerCount;
        this.espeakVoice = espeakVoice;
        this.phonemeIds = phonemeIds;
        this.noiseScale = noiseScale;
        this.lengthScale = lengthScale;
        this.noiseScaleW = noiseScaleW;
    }

    public static PiperVoiceConfig load(Path configFile) throws IOException {
        Objects.requireNonNull(configFile, "configFile");
        if (!Files.isRegularFile(configFile)) {
            throw new IOException("Missing Piper voice configuration: " + configFile);
        }
        JsonNode root = new ObjectMapper().readTree(configFile.toFile());
        require(root != null && root.isObject(), "Invalid Piper voice configuration");
        require("espeak".equals(text(root, "phoneme_type")), "Only eSpeak Piper phonemes are supported");
        JsonNode audio = root.path("audio");
        int sampleRate = positiveInt(audio.path("sample_rate"), "audio.sample_rate");
        int speakers = positiveInt(root.path("num_speakers"), "num_speakers");
        String voice = text(root.path("espeak"), "voice");
        require(voice != null && !voice.isBlank(), "Missing espeak.voice");
        JsonNode parameters = root.path("inference");
        float noise = nonNegativeFloat(parameters.path("noise_scale"), "inference.noise_scale");
        float length = positiveFloat(parameters.path("length_scale"), "inference.length_scale");
        float noiseW = nonNegativeFloat(parameters.path("noise_w"), "inference.noise_w");
        JsonNode mapping = root.path("phoneme_id_map");
        require(mapping.isObject() && !mapping.isEmpty(), "Missing phoneme_id_map");
        Map<String, int[]> ids = new LinkedHashMap<>();
        mapping.fields().forEachRemaining(entry -> {
            JsonNode values = entry.getValue();
            if (entry.getKey().isEmpty() || !values.isArray() || values.isEmpty()) {
                throw new IllegalArgumentException("Invalid Piper phoneme ID mapping: " + entry.getKey());
            }
            int[] mapped = new int[values.size()];
            for (int index = 0; index < values.size(); index++) {
                JsonNode value = values.get(index);
                if (!value.canConvertToInt() || value.intValue() < 0) {
                    throw new IllegalArgumentException("Invalid Piper phoneme ID for " + entry.getKey());
                }
                mapped[index] = value.intValue();
            }
            ids.put(entry.getKey(), mapped);
        });
        for (String special : new String[]{"^", "_", "$"}) {
            require(ids.containsKey(special), "Missing Piper phoneme marker " + special);
        }
        return new PiperVoiceConfig(sampleRate, speakers, voice,
                Collections.unmodifiableMap(ids), noise, length, noiseW);
    }

    public int sampleRate() {
        return sampleRate;
    }

    public int speakerCount() {
        return speakerCount;
    }

    public String espeakVoice() {
        return espeakVoice;
    }

    public float noiseScale() {
        return noiseScale;
    }

    public float lengthScale() {
        return lengthScale;
    }

    public float noiseScaleW() {
        return noiseScaleW;
    }

    public int[] idsFor(String phoneme) {
        int[] ids = phonemeIds.get(phoneme);
        return ids == null ? null : ids.clone();
    }

    public int phonemeCount() {
        return phonemeIds.size();
    }

    private static String text(JsonNode parent, String name) {
        JsonNode value = parent.path(name);
        return value.isTextual() ? value.textValue() : null;
    }

    private static int positiveInt(JsonNode node, String property) throws IOException {
        if (!node.canConvertToInt() || node.intValue() < 1) {
            throw new IOException("Invalid Piper " + property);
        }
        return node.intValue();
    }

    private static float nonNegativeFloat(JsonNode node, String property) throws IOException {
        if (!node.isNumber() || !Float.isFinite(node.floatValue()) || node.floatValue() < 0) {
            throw new IOException("Invalid Piper " + property);
        }
        return node.floatValue();
    }

    private static float positiveFloat(JsonNode node, String property) throws IOException {
        float value = nonNegativeFloat(node, property);
        if (value == 0) {
            throw new IOException("Invalid Piper " + property);
        }
        return value;
    }

    private static void require(boolean condition, String message) throws IOException {
        if (!condition) {
            throw new IOException(message);
        }
    }
}
