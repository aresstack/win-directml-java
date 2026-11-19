package com.aresstack.windirectml.inference.whisper;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Decode Whisper multilingual byte-level BPE token IDs into UTF-8 text, and encode text
 * into the same IDs for decoder prompts. Read the vocabulary, merge ranks and special
 * token mapping already distributed by Hugging Face.
 */
public final class WhisperTokenizer {
    /** The GPT-2 pre-tokenization pattern used by Hugging Face's byte-level BPE. */
    private static final Pattern PRE_TOKENIZER = Pattern.compile(
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

    private final String[] pieces;
    private final Map<Integer, Integer> byteByCodePoint;
    private final Map<String, Integer> specialTokens;
    private final Map<String, Integer> idByPiece = new HashMap<>();
    private final Map<String, Integer> mergeRanks = new HashMap<>();
    private final char[] codePointByByte = new char[256];

    public WhisperTokenizer(Path modelDirectory) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        JsonNode vocabulary = mapper.readTree(modelDirectory.resolve("vocab.json").toFile());
        pieces = new String[50257];
        var tokens = vocabulary.fields();
        while (tokens.hasNext()) {
            var entry = tokens.next();
            int token = entry.getValue().asInt();
            if (token >= 0 && token < pieces.length) {
                pieces[token] = entry.getKey();
                idByPiece.put(entry.getKey(), token);
            }
        }
        byteByCodePoint = createByteDecoder();
        for (Map.Entry<Integer, Integer> mapping : byteByCodePoint.entrySet()) {
            codePointByByte[mapping.getValue()] = (char) (int) mapping.getKey();
        }
        specialTokens = new HashMap<>();
        JsonNode tokenizer = mapper.readTree(modelDirectory.resolve("tokenizer.json").toFile());
        JsonNode merges = tokenizer.path("model").path("merges");
        for (int rank = 0; rank < merges.size(); rank++) {
            JsonNode merge = merges.get(rank);
            String pair = merge.isArray() ? merge.get(0).asText() + " " + merge.get(1).asText() : merge.asText();
            mergeRanks.putIfAbsent(pair, rank);
        }
        JsonNode addedTokens = tokenizer.path("added_tokens");
        for (JsonNode added : addedTokens) {
            specialTokens.put(added.path("content").asText(), added.path("id").asInt());
        }
    }

    public int requireSpecial(String name) {
        Integer token = specialTokens.get(name);
        if (token == null) throw new IllegalArgumentException("Unknown Whisper special token: " + name);
        return token;
    }

    /** Return the ID of the first special token that exists, or -1 if none does. */
    public int findSpecial(String... names) {
        for (String name : names) {
            Integer token = specialTokens.get(name);
            if (token != null) {
                return token;
            }
        }
        return -1;
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

    /**
     * Encode plain text into text-token IDs exactly like the Hugging Face byte-level BPE
     * tokenizer of the checkpoint. Special tokens are never produced.
     */
    public List<Integer> encode(String text) {
        List<Integer> ids = new ArrayList<>();
        Matcher words = PRE_TOKENIZER.matcher(text);
        while (words.find()) {
            byte[] bytes = words.group().getBytes(StandardCharsets.UTF_8);
            List<String> symbols = new ArrayList<>(bytes.length);
            for (byte value : bytes) {
                symbols.add(String.valueOf(codePointByByte[value & 0xff]));
            }
            for (String piece : mergeAll(symbols)) {
                Integer id = idByPiece.get(piece);
                if (id == null) {
                    throw new IllegalArgumentException("Whisper vocabulary lacks BPE piece: " + piece);
                }
                ids.add(id);
            }
        }
        return ids;
    }

    /** Repeatedly merge the adjacent pair with the lowest rank until no ranked pair remains. */
    private List<String> mergeAll(List<String> symbols) {
        while (symbols.size() > 1) {
            int best = -1;
            int bestRank = Integer.MAX_VALUE;
            for (int i = 0; i + 1 < symbols.size(); i++) {
                Integer rank = mergeRanks.get(symbols.get(i) + " " + symbols.get(i + 1));
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    best = i;
                }
            }
            if (best < 0) {
                break;
            }
            String first = symbols.get(best);
            String second = symbols.get(best + 1);
            List<String> merged = new ArrayList<>(symbols.size() - 1);
            for (int i = 0; i < symbols.size(); i++) {
                if (i + 1 < symbols.size() && symbols.get(i).equals(first) && symbols.get(i + 1).equals(second)) {
                    merged.add(first + second);
                    i++;
                } else {
                    merged.add(symbols.get(i));
                }
            }
            symbols = merged;
        }
        return symbols;
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
