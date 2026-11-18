package com.aresstack.windirectml.inference.whisper;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Decode Whisper multilingual byte-level BPE token IDs into UTF-8 text.
 * Read the vocabulary and special token mapping already distributed by Hugging Face.
 */
public final class WhisperTokenizer {
    private final String[] pieces;
    private final Map<Integer, Integer> byteByCodePoint;
    private final Map<String, Integer> specialTokens;

    public WhisperTokenizer(Path modelDirectory) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        JsonNode vocabulary = mapper.readTree(modelDirectory.resolve("vocab.json").toFile());
        pieces = new String[50257];
        var tokens = vocabulary.fields();
        while (tokens.hasNext()) {
            var entry = tokens.next();
            int token = entry.getValue().asInt();
            if (token >= 0 && token < pieces.length) pieces[token] = entry.getKey();
        }
        byteByCodePoint = createByteDecoder();
        specialTokens = new HashMap<>();
        JsonNode addedTokens = mapper.readTree(modelDirectory.resolve("tokenizer.json").toFile())
                .path("added_tokens");
        for (JsonNode added : addedTokens) {
            specialTokens.put(added.path("content").asText(), added.path("id").asInt());
        }
    }

    public int requireSpecial(String name) {
        Integer token = specialTokens.get(name);
        if (token == null) throw new IllegalArgumentException("Unknown Whisper special token: " + name);
        return token;
    }

    public int textTokenCount() {
        return pieces.length;
    }

    public String decode(List<Integer> ids) {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        for (Integer id : ids) {
            if (id == null || id < 0 || id >= pieces.length) continue;
            String piece = pieces[id];
            if (piece == null) throw new IllegalArgumentException("Unmapped Whisper vocabulary ID: " + id);
            piece.codePoints().forEach(codePoint -> {
                Integer value = byteByCodePoint.get(codePoint);
                if (value == null) throw new IllegalArgumentException("Unknown Whisper byte mapping: " + codePoint);
                output.write(value);
            });
        }
        return new String(output.toByteArray(), StandardCharsets.UTF_8);
    }

    private static Map<Integer, Integer> createByteDecoder() {
        Map<Integer, Integer> mapping = new HashMap<>();
        int extra = 0;
        for (int value = 0; value <= 255; value++) {
            int glyph;
            if ((value >= 33 && value <= 126)
                    || (value >= 161 && value <= 172)
                    || (value >= 174 && value <= 255)) {
                glyph = value;
            } else {
                glyph = 256 + extra++;
            }
            mapping.put(glyph, value);
        }
        return mapping;
    }
}
