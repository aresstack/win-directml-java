package com.aresstack.windirectml.inference.t5;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Lightweight tokenizer.json based fallback for google-t5 style SentencePiece tokenizers.
 *
 * <p>This class intentionally avoids native SentencePiece dependencies. It is sufficient for
 * loading and smoke-testing upstream T5 SafeTensors packages on hardened systems. A later patch
 * can replace the segmentation strategy with a full SentencePiece implementation without
 * changing {@link T5InferenceEngine}.</p>
 */
final class JsonT5Tokenizer implements T5TextTokenizer {
    private static final String WORD_PREFIX = "▁";

    private final Map<String, Integer> tokenToId;
    private final String[] idToToken;
    private final int unknownTokenId;
    private final int eosTokenId;
    private final int padTokenId;

    private JsonT5Tokenizer(Map<String, Integer> tokenToId,
                            String[] idToToken,
                            int unknownTokenId,
                            int eosTokenId,
                            int padTokenId) {
        this.tokenToId = Map.copyOf(new LinkedHashMap<String, Integer>(tokenToId));
        this.idToToken = idToToken.clone();
        this.unknownTokenId = unknownTokenId;
        this.eosTokenId = eosTokenId;
        this.padTokenId = padTokenId;
    }

    static JsonT5Tokenizer load(Path modelDir) throws IOException {
        Objects.requireNonNull(modelDir, "modelDir");
        Path tokenizerJson = modelDir.resolve("tokenizer.json");
        if (!Files.isRegularFile(tokenizerJson)) {
            throw new IOException("T5 tokenizer.json not found: " + tokenizerJson);
        }
        JsonNode root = new ObjectMapper().readTree(tokenizerJson.toFile());
        Map<String, Integer> tokenToId = readVocabulary(root);
        if (tokenToId.isEmpty()) {
            throw new IOException("T5 tokenizer.json does not contain a readable model.vocab section: " + tokenizerJson);
        }
        String[] idToToken = reverse(tokenToId);
        int unknown = tokenToId.getOrDefault("<unk>", 2);
        int eos = tokenToId.getOrDefault("</s>", 1);
        int pad = tokenToId.getOrDefault("<pad>", 0);
        return new JsonT5Tokenizer(tokenToId, idToToken, unknown, eos, pad);
    }

    @Override
    public int[] encode(String text) {
        if (text == null || text.isBlank()) {
            return new int[]{eosTokenId};
        }
        String normalized = Normalizer.normalize(text, Normalizer.Form.NFKC);
        List<Integer> out = new ArrayList<Integer>();
        for (String word : normalized.trim().split("\\s+")) {
            encodeWord(word, out);
        }
        out.add(eosTokenId);
        return toIntArray(out);
    }

    @Override
    public String decode(int[] tokenIds) {
        if (tokenIds == null || tokenIds.length == 0) {
            return "";
        }
        StringBuilder out = new StringBuilder();
        for (int id : tokenIds) {
            if (id == eosTokenId || id == padTokenId || id < 0 || id >= idToToken.length) {
                continue;
            }
            String token = idToToken[id];
            if (token == null || token.isEmpty() || "<unk>".equals(token)) {
                continue;
            }
            if (token.startsWith(WORD_PREFIX)) {
                if (out.length() > 0) {
                    out.append(' ');
                }
                out.append(token.substring(WORD_PREFIX.length()));
            } else {
                out.append(token);
            }
        }
        return out.toString().trim();
    }

    @Override
    public int vocabSize() {
        return idToToken.length;
    }

    private void encodeWord(String word, List<Integer> out) {
        if (word.isEmpty()) {
            return;
        }
        String prefixed = WORD_PREFIX + word;
        Integer direct = tokenToId.get(prefixed);
        if (direct != null) {
            out.add(direct);
            return;
        }
        direct = tokenToId.get(word);
        if (direct != null) {
            out.add(direct);
            return;
        }
        encodeGreedyPieces(prefixed, out);
    }

    private void encodeGreedyPieces(String text, List<Integer> out) {
        int index = 0;
        while (index < text.length()) {
            Match match = longestMatch(text, index);
            if (match.length <= 0) {
                out.add(unknownTokenId);
                index += Character.charCount(text.codePointAt(index));
            } else {
                out.add(match.tokenId);
                index += match.length;
            }
        }
    }

    private Match longestMatch(String text, int index) {
        int maxEnd = text.length();
        for (int end = maxEnd; end > index; end--) {
            String candidate = text.substring(index, end);
            Integer id = tokenToId.get(candidate);
            if (id != null) {
                return new Match(id, end - index);
            }
        }
        return new Match(unknownTokenId, 0);
    }

    private static Map<String, Integer> readVocabulary(JsonNode root) throws IOException {
        JsonNode vocab = root.path("model").path("vocab");
        if (!vocab.isArray()) {
            return Map.of();
        }
        LinkedHashMap<String, Integer> out = new LinkedHashMap<String, Integer>();
        int nextId = 0;
        for (JsonNode item : vocab) {
            if (item.isArray() && item.size() > 0) {
                out.put(item.get(0).asText(), nextId++);
            } else if (item.isTextual()) {
                out.put(item.asText(), nextId++);
            }
        }
        return out;
    }

    private static String[] reverse(Map<String, Integer> vocab) throws IOException {
        int max = -1;
        for (Integer id : vocab.values()) {
            if (id == null || id < 0) {
                throw new IOException("T5 tokenizer contains invalid token id: " + id);
            }
            max = Math.max(max, id);
        }
        String[] out = new String[max + 1];
        for (Map.Entry<String, Integer> entry : vocab.entrySet()) {
            out[entry.getValue()] = entry.getKey();
        }
        return out;
    }

    private static int[] toIntArray(List<Integer> values) {
        int[] out = new int[values.size()];
        for (int i = 0; i < values.size(); i++) {
            out[i] = values.get(i);
        }
        return out;
    }

    private record Match(int tokenId, int length) {
    }
}
