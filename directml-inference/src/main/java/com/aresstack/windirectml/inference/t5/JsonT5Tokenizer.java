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
import java.util.Map;
import java.util.Objects;

/**
 * Tokenizer.json based SentencePiece-Unigram tokenizer for google-t5 style models.
 *
 * <p>The implementation intentionally stays in Java and does not load a native
 * SentencePiece runtime. It reads the Unigram vocabulary and scores from
 * {@code tokenizer.json}, applies a Viterbi segmentation over the SentencePiece
 * word-boundary representation, and keeps the external tokenizer boundary stable
 * for the T5 inference engine.</p>
 */
final class JsonT5Tokenizer implements T5TextTokenizer {
    private static final String WORD_PREFIX = "▁";
    private static final float UNKNOWN_SCORE = -100.0f;

    private final Map<String, Integer> tokenToId;
    private final Map<String, Float> tokenScores;
    private final String[] idToToken;
    private final int unknownTokenId;
    private final int eosTokenId;
    private final int padTokenId;
    private final int maxTokenLength;

    private JsonT5Tokenizer(Map<String, Integer> tokenToId,
                            Map<String, Float> tokenScores,
                            String[] idToToken,
                            int unknownTokenId,
                            int eosTokenId,
                            int padTokenId,
                            int maxTokenLength) {
        this.tokenToId = Map.copyOf(new LinkedHashMap<String, Integer>(tokenToId));
        this.tokenScores = Map.copyOf(new LinkedHashMap<String, Float>(tokenScores));
        this.idToToken = idToToken.clone();
        this.unknownTokenId = unknownTokenId;
        this.eosTokenId = eosTokenId;
        this.padTokenId = padTokenId;
        this.maxTokenLength = Math.max(1, maxTokenLength);
    }

    static JsonT5Tokenizer load(Path modelDir) throws IOException {
        Objects.requireNonNull(modelDir, "modelDir");
        Path tokenizerJson = modelDir.resolve("tokenizer.json");
        if (!Files.isRegularFile(tokenizerJson)) {
            throw new IOException("T5 tokenizer.json not found: " + tokenizerJson);
        }
        JsonNode root = new ObjectMapper().readTree(tokenizerJson.toFile());
        Vocabulary vocabulary = readVocabulary(root);
        if (vocabulary.tokenToId.isEmpty()) {
            throw new IOException("T5 tokenizer.json does not contain a readable model.vocab section: " + tokenizerJson);
        }
        String[] idToToken = reverse(vocabulary.tokenToId);
        int unknown = vocabulary.tokenToId.getOrDefault("<unk>", 2);
        int eos = vocabulary.tokenToId.getOrDefault("</s>", 1);
        int pad = vocabulary.tokenToId.getOrDefault("<pad>", 0);
        return new JsonT5Tokenizer(vocabulary.tokenToId, vocabulary.tokenScores, idToToken,
                unknown, eos, pad, vocabulary.maxTokenLength);
    }

    @Override
    public int[] encode(String text) {
        if (text == null || text.isBlank()) {
            return new int[]{eosTokenId};
        }
        String sentencePieceText = toSentencePieceText(text);
        List<Integer> out = encodeUnigram(sentencePieceText);
        out.add(eosTokenId);
        return toIntArray(out);
    }

    @Override
    public String decode(int[] tokenIds) {
        if (tokenIds == null || tokenIds.length == 0) {
            return "";
        }
        StringBuilder pieces = new StringBuilder();
        for (int id : tokenIds) {
            if (id == eosTokenId || id == padTokenId || id < 0 || id >= idToToken.length) {
                continue;
            }
            String token = idToToken[id];
            if (token == null || token.isEmpty() || "<unk>".equals(token)) {
                continue;
            }
            pieces.append(token);
        }
        return pieces.toString()
                .replace(WORD_PREFIX, " ")
                .replaceAll("\\s+", " ")
                .trim();
    }

    @Override
    public int vocabSize() {
        return idToToken.length;
    }

    @Override
    public String describeToken(int tokenId) {
        if (tokenId < 0 || tokenId >= idToToken.length) {
            return tokenId + ":<out-of-range>";
        }
        String token = idToToken[tokenId];
        if (token == null || token.isEmpty()) {
            return tokenId + ":<empty>";
        }
        return tokenId + ":" + token;
    }

    private List<Integer> encodeUnigram(String text) {
        int length = text.length();
        float[] bestScore = new float[length + 1];
        int[] bestStart = new int[length + 1];
        int[] bestToken = new int[length + 1];
        Arrays.fill(bestScore, Float.NEGATIVE_INFINITY);
        Arrays.fill(bestStart, -1);
        Arrays.fill(bestToken, unknownTokenId);
        bestScore[0] = 0.0f;

        for (int start = 0; start < length; start++) {
            if (bestStart[start] < 0 && start != 0) {
                continue;
            }
            int maxEnd = Math.min(length, start + maxTokenLength);
            boolean matched = false;
            for (int end = start + 1; end <= maxEnd; end++) {
                String candidate = text.substring(start, end);
                Integer tokenId = tokenToId.get(candidate);
                if (tokenId == null) {
                    continue;
                }
                matched = true;
                float score = bestScore[start] + score(candidate);
                if (score > bestScore[end]) {
                    bestScore[end] = score;
                    bestStart[end] = start;
                    bestToken[end] = tokenId;
                }
            }
            if (!matched) {
                int end = start + Character.charCount(text.codePointAt(start));
                float score = bestScore[start] + UNKNOWN_SCORE;
                if (score > bestScore[end]) {
                    bestScore[end] = score;
                    bestStart[end] = start;
                    bestToken[end] = unknownTokenId;
                }
            }
        }

        if (bestStart[length] < 0) {
            return List.of(unknownTokenId);
        }
        ArrayList<Integer> reversed = new ArrayList<Integer>();
        int index = length;
        while (index > 0) {
            reversed.add(bestToken[index]);
            index = bestStart[index];
        }
        ArrayList<Integer> out = new ArrayList<Integer>(reversed.size());
        for (int i = reversed.size() - 1; i >= 0; i--) {
            out.add(reversed.get(i));
        }
        return out;
    }

    private float score(String token) {
        Float score = tokenScores.get(token);
        return score == null ? UNKNOWN_SCORE : score;
    }

    private static String toSentencePieceText(String text) {
        String normalized = Normalizer.normalize(text, Normalizer.Form.NFKC).trim();
        StringBuilder out = new StringBuilder(normalized.length() + 1);
        boolean pendingWordPrefix = true;
        for (int offset = 0; offset < normalized.length(); ) {
            int codePoint = normalized.codePointAt(offset);
            offset += Character.charCount(codePoint);
            if (Character.isWhitespace(codePoint)) {
                pendingWordPrefix = true;
                continue;
            }
            if (pendingWordPrefix) {
                out.append(WORD_PREFIX);
                pendingWordPrefix = false;
            }
            out.appendCodePoint(codePoint);
        }
        return out.toString();
    }

    private static Vocabulary readVocabulary(JsonNode root) throws IOException {
        JsonNode vocab = root.path("model").path("vocab");
        if (!vocab.isArray()) {
            return Vocabulary.empty();
        }
        LinkedHashMap<String, Integer> tokenToId = new LinkedHashMap<String, Integer>();
        LinkedHashMap<String, Float> tokenScores = new LinkedHashMap<String, Float>();
        int nextId = 0;
        int maxTokenLength = 1;
        for (JsonNode item : vocab) {
            String token;
            float score;
            if (item.isArray() && item.size() > 0) {
                token = item.get(0).asText();
                score = item.size() > 1 && item.get(1).isNumber() ? (float) item.get(1).asDouble() : 0.0f;
            } else if (item.isTextual()) {
                token = item.asText();
                score = 0.0f;
            } else {
                continue;
            }
            tokenToId.put(token, nextId++);
            tokenScores.put(token, score);
            maxTokenLength = Math.max(maxTokenLength, token.length());
        }
        return new Vocabulary(tokenToId, tokenScores, maxTokenLength);
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

    private static final class Vocabulary {
        private final Map<String, Integer> tokenToId;
        private final Map<String, Float> tokenScores;
        private final int maxTokenLength;

        private Vocabulary(Map<String, Integer> tokenToId,
                           Map<String, Float> tokenScores,
                           int maxTokenLength) {
            this.tokenToId = tokenToId;
            this.tokenScores = tokenScores;
            this.maxTokenLength = maxTokenLength;
        }

        private static Vocabulary empty() {
            return new Vocabulary(Map.of(), Map.of(), 1);
        }
    }
}
