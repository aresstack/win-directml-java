package com.aresstack.windirectml.inference.smollm2;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Byte-level BPE tokenizer for SmolLM2 tokenizer.json files.
 *
 * <p>This tokenizer is intentionally scoped to the local SmolLM2 runtime bring-up. It supports the tokenizer.json
 * fields needed by Hugging Face byte-level BPE tokenizers: {@code model.vocab}, {@code model.merges} and
 * {@code added_tokens}. It keeps chat-template rendering outside of the tokenizer so runtime tests can use raw text
 * prompts without pulling in UI or provider-specific prompt policy.</p>
 */
public final class SmolLM2Tokenizer {

    private static final char[] BYTE_TO_UNICODE = new char[256];
    private static final Map<Character, Byte> UNICODE_TO_BYTE = new HashMap<>(512);

    private static final Pattern PRE_TOKENIZE_PATTERN = Pattern.compile(
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    );

    static {
        int shifted = 0;
        for (int b = 0; b < 256; b++) {
            if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) {
                BYTE_TO_UNICODE[b] = (char) b;
            } else {
                BYTE_TO_UNICODE[b] = (char) (256 + shifted);
                shifted++;
            }
        }
        for (int b = 0; b < 256; b++) {
            UNICODE_TO_BYTE.put(BYTE_TO_UNICODE[b], (byte) b);
        }
    }

    private final Map<String, Integer> vocab;
    private final String[] idToToken;
    private final Map<Long, Integer> mergeRanks;
    private final Map<String, Integer> specialTokens;

    private SmolLM2Tokenizer(Map<String, Integer> vocab,
                             String[] idToToken,
                             Map<Long, Integer> mergeRanks,
                             Map<String, Integer> specialTokens) {
        this.vocab = Map.copyOf(Objects.requireNonNull(vocab, "vocab"));
        this.idToToken = Objects.requireNonNull(idToToken, "idToToken").clone();
        this.mergeRanks = Map.copyOf(Objects.requireNonNull(mergeRanks, "mergeRanks"));
        this.specialTokens = Map.copyOf(Objects.requireNonNull(specialTokens, "specialTokens"));
    }

    public static SmolLM2Tokenizer load(Path tokenizerJson) throws IOException {
        Objects.requireNonNull(tokenizerJson, "tokenizerJson");
        ObjectMapper mapper = new ObjectMapper();
        JsonNode root = mapper.readTree(tokenizerJson.toFile());
        JsonNode model = requireObject(root, "model");
        Map<String, Integer> vocab = readVocab(model);
        Map<String, Integer> specialTokens = readSpecialTokens(root, vocab);
        String[] idToToken = createIdToToken(vocab);
        Map<Long, Integer> mergeRanks = readMerges(model);
        return new SmolLM2Tokenizer(vocab, idToToken, mergeRanks, specialTokens);
    }

    public int[] encode(String text) {
        if (text == null || text.isEmpty()) {
            return new int[0];
        }
        List<Integer> ids = new ArrayList<>();
        for (Object segment : splitSpecialTokens(text)) {
            if (segment instanceof Integer) {
                ids.add((Integer) segment);
            } else {
                encodePlainText((String) segment, ids);
            }
        }
        return ids.stream().mapToInt(Integer::intValue).toArray();
    }

    public String decode(int[] ids) {
        return decode(ids, false);
    }

    public String decode(int[] ids, boolean skipSpecialTokens) {
        Objects.requireNonNull(ids, "ids");
        StringBuilder tokenChars = new StringBuilder();
        for (int id : ids) {
            if (id < 0 || id >= idToToken.length) {
                continue;
            }
            String token = idToToken[id];
            if (skipSpecialTokens && specialTokens.containsKey(token)) {
                continue;
            }
            tokenChars.append(token);
        }
        return decodeByteLevelString(tokenChars.toString());
    }

    public String decodeToken(int id) {
        if (id < 0 || id >= idToToken.length) {
            return "";
        }
        return idToToken[id];
    }

    public int vocabSize() {
        return idToToken.length;
    }

    public Map<String, Integer> specialTokens() {
        return Collections.unmodifiableMap(specialTokens);
    }

    public boolean isSpecialToken(int tokenId) {
        if (tokenId < 0 || tokenId >= idToToken.length) {
            return false;
        }
        return specialTokens.containsKey(idToToken[tokenId]);
    }

    private void encodePlainText(String text, List<Integer> ids) {
        Matcher matcher = PRE_TOKENIZE_PATTERN.matcher(text);
        while (matcher.find()) {
            encodeBpeToken(matcher.group(), ids);
        }
    }

    private void encodeBpeToken(String preToken, List<Integer> ids) {
        byte[] bytes = preToken.getBytes(StandardCharsets.UTF_8);
        List<String> symbols = new ArrayList<>(bytes.length);
        for (byte value : bytes) {
            symbols.add(String.valueOf(BYTE_TO_UNICODE[value & 0xFF]));
        }
        applyMerges(symbols);
        for (String symbol : symbols) {
            Integer id = vocab.get(symbol);
            if (id != null) {
                ids.add(id);
            } else {
                addByteFallbackTokens(symbol, ids);
            }
        }
    }

    private void applyMerges(List<String> symbols) {
        while (symbols.size() > 1) {
            int bestIndex = -1;
            int bestRank = Integer.MAX_VALUE;
            for (int i = 0; i < symbols.size() - 1; i++) {
                Integer rank = mergeRanks.get(pairKey(symbols.get(i), symbols.get(i + 1)));
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestIndex = i;
                }
            }
            if (bestIndex < 0) {
                return;
            }
            String merged = symbols.get(bestIndex) + symbols.get(bestIndex + 1);
            symbols.set(bestIndex, merged);
            symbols.remove(bestIndex + 1);
        }
    }

    private void addByteFallbackTokens(String symbol, List<Integer> ids) {
        for (int i = 0; i < symbol.length(); i++) {
            Integer id = vocab.get(String.valueOf(symbol.charAt(i)));
            if (id != null) {
                ids.add(id);
            }
        }
    }

    private List<Object> splitSpecialTokens(String text) {
        List<Object> result = new ArrayList<>();
        List<Map.Entry<String, Integer>> tokens = new ArrayList<>(specialTokens.entrySet());
        tokens.sort((left, right) -> Integer.compare(right.getKey().length(), left.getKey().length()));
        splitSpecialTokens(text, tokens, result);
        return result;
    }

    private void splitSpecialTokens(String text, List<Map.Entry<String, Integer>> tokens, List<Object> result) {
        if (text.isEmpty()) {
            return;
        }
        int bestPosition = text.length();
        Map.Entry<String, Integer> bestToken = null;
        for (Map.Entry<String, Integer> token : tokens) {
            int position = text.indexOf(token.getKey());
            if (position >= 0 && position < bestPosition) {
                bestPosition = position;
                bestToken = token;
            }
        }
        if (bestToken == null) {
            result.add(text);
            return;
        }
        if (bestPosition > 0) {
            result.add(text.substring(0, bestPosition));
        }
        result.add(bestToken.getValue());
        splitSpecialTokens(text.substring(bestPosition + bestToken.getKey().length()), tokens, result);
    }

    private String decodeByteLevelString(String tokenChars) {
        char[] chars = tokenChars.toCharArray();
        byte[] bytes = new byte[chars.length];
        int byteCount = 0;
        for (char c : chars) {
            Byte value = UNICODE_TO_BYTE.get(c);
            if (value != null) {
                bytes[byteCount] = value;
                byteCount++;
            }
        }
        return new String(bytes, 0, byteCount, StandardCharsets.UTF_8);
    }

    private static Map<String, Integer> readVocab(JsonNode model) throws IOException {
        JsonNode vocabNode = requireObject(model, "vocab");
        Map<String, Integer> vocab = new LinkedHashMap<>();
        vocabNode.fields().forEachRemaining(entry -> vocab.put(entry.getKey(), entry.getValue().asInt()));
        return vocab;
    }

    private static Map<String, Integer> readSpecialTokens(JsonNode root, Map<String, Integer> vocab) {
        Map<String, Integer> specialTokens = new LinkedHashMap<>();
        JsonNode addedTokens = root.get("added_tokens");
        if (addedTokens == null || !addedTokens.isArray()) {
            return specialTokens;
        }
        for (JsonNode token : addedTokens) {
            if (!token.has("content") || !token.has("id")) {
                continue;
            }
            String content = token.get("content").asText();
            int id = token.get("id").asInt();
            boolean special = token.has("special") && token.get("special").asBoolean();
            vocab.put(content, id);
            if (special) {
                specialTokens.put(content, id);
            }
        }
        return specialTokens;
    }

    private static String[] createIdToToken(Map<String, Integer> vocab) {
        int maxId = vocab.values().stream().mapToInt(Integer::intValue).max().orElse(0);
        String[] idToToken = new String[maxId + 1];
        Arrays.fill(idToToken, "");
        for (Map.Entry<String, Integer> entry : vocab.entrySet()) {
            int id = entry.getValue();
            if (id >= 0 && id < idToToken.length) {
                idToToken[id] = entry.getKey();
            }
        }
        return idToToken;
    }

    private static Map<Long, Integer> readMerges(JsonNode model) throws IOException {
        JsonNode mergesNode = model.get("merges");
        Map<Long, Integer> mergeRanks = new HashMap<>();
        if (mergesNode == null || !mergesNode.isArray()) {
            return mergeRanks;
        }
        for (int i = 0; i < mergesNode.size(); i++) {
            String[] pair = readMergePair(mergesNode.get(i), i);
            mergeRanks.put(pairKey(pair[0], pair[1]), i);
        }
        return mergeRanks;
    }

    private static String[] readMergePair(JsonNode entry, int index) throws IOException {
        if (entry.isArray() && entry.size() == 2) {
            return new String[]{entry.get(0).asText(), entry.get(1).asText()};
        }
        String merge = entry.asText();
        int separator = merge.indexOf(' ');
        if (separator < 0) {
            throw new IOException("Invalid SmolLM2 merge entry at index " + index + ": " + merge);
        }
        return new String[]{merge.substring(0, separator), merge.substring(separator + 1)};
    }

    private static JsonNode requireObject(JsonNode parent, String fieldName) throws IOException {
        JsonNode node = parent == null ? null : parent.get(fieldName);
        if (node == null || !node.isObject()) {
            throw new IOException("Invalid SmolLM2 tokenizer.json: missing object field " + fieldName);
        }
        return node;
    }

    private static long pairKey(String left, String right) {
        return ((long) left.hashCode()) << 32 | (right.hashCode() & 0xFFFFFFFFL);
    }
}
