package com.aresstack.windirectml.inference.t5;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Byte-level BPE tokenizer for the curated Salesforce CodeT5 checkpoints.
 *
 * <p>CodeT5 uses a RoBERTa-style tokenizer with {@code vocab.json} and
 * {@code merges.txt}. This implementation is intentionally small and local to
 * the T5 family. It is used by the Workbench and {@link T5InferenceEngine} to
 * bridge user text to the token-id based T5 runtime without involving Python,
 * Hugging Face tokenizers, or native tokenizer libraries.</p>
 */
public final class CodeT5Tokenizer {
    private static final Pattern PRE_TOKENIZE_PATTERN = Pattern.compile(
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    );

    private static final char[] BYTE_TO_UNICODE = new char[256];
    private static final Map<Character, Byte> UNICODE_TO_BYTE = new HashMap<>(512);

    static {
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) {
                BYTE_TO_UNICODE[b] = (char) b;
            } else {
                BYTE_TO_UNICODE[b] = (char) (256 + n);
                n++;
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
    private final int unknownTokenId;

    private CodeT5Tokenizer(Map<String, Integer> vocab,
                            String[] idToToken,
                            Map<Long, Integer> mergeRanks,
                            Map<String, Integer> specialTokens,
                            int unknownTokenId) {
        this.vocab = Map.copyOf(new LinkedHashMap<>(vocab));
        this.idToToken = idToToken.clone();
        this.mergeRanks = Map.copyOf(new LinkedHashMap<>(mergeRanks));
        this.specialTokens = Map.copyOf(new LinkedHashMap<>(specialTokens));
        this.unknownTokenId = unknownTokenId;
    }

    /**
     * Load the CodeT5 tokenizer files from a Hugging Face model directory.
     *
     * @param modelDir directory containing {@code vocab.json} and {@code merges.txt}
     * @return tokenizer instance
     * @throws IOException if required files are missing or invalid
     */
    public static CodeT5Tokenizer load(Path modelDir) throws IOException {
        Path root = Objects.requireNonNull(modelDir, "modelDir").toAbsolutePath().normalize();
        Path vocabJson = root.resolve("vocab.json");
        Path mergesTxt = root.resolve("merges.txt");
        if (!Files.isRegularFile(vocabJson)) {
            throw new IOException("CodeT5 vocab.json not found: " + vocabJson);
        }
        if (!Files.isRegularFile(mergesTxt)) {
            throw new IOException("CodeT5 merges.txt not found: " + mergesTxt);
        }
        Map<String, Integer> vocab = readVocab(vocabJson);
        String[] idToToken = reverse(vocab);
        Map<String, Integer> specialTokens = readSpecialTokens(root, vocab);
        Map<Long, Integer> mergeRanks = readMerges(mergesTxt);
        int unknownTokenId = specialTokens.getOrDefault("<unk>", vocab.getOrDefault("<unk>", -1));
        if (unknownTokenId < 0) {
            throw new IOException("CodeT5 tokenizer has no <unk> token in vocab/special tokens");
        }
        return new CodeT5Tokenizer(vocab, idToToken, mergeRanks, specialTokens, unknownTokenId);
    }

    public int[] encode(String text) {
        if (text == null || text.isEmpty()) {
            return new int[0];
        }
        List<Integer> out = new ArrayList<>();
        for (Object segment : splitSpecialTokens(text)) {
            if (segment instanceof Integer id) {
                out.add(id);
            } else {
                encodeRegularText((String) segment, out);
            }
        }
        return toIntArray(out);
    }

    public String decode(int[] tokenIds) {
        if (tokenIds == null || tokenIds.length == 0) {
            return "";
        }
        StringBuilder tokenText = new StringBuilder();
        for (int tokenId : tokenIds) {
            if (tokenId < 0 || tokenId >= idToToken.length) {
                continue;
            }
            String token = idToToken[tokenId];
            if (token == null || token.isEmpty() || specialTokens.containsKey(token)) {
                continue;
            }
            tokenText.append(token);
        }
        return decodeByteLevelText(tokenText.toString());
    }

    public int vocabSize() {
        return idToToken.length;
    }

    public int unknownTokenId() {
        return unknownTokenId;
    }

    public Map<String, Integer> specialTokens() {
        return specialTokens;
    }

    private void encodeRegularText(String text, List<Integer> out) {
        Matcher matcher = PRE_TOKENIZE_PATTERN.matcher(text);
        while (matcher.find()) {
            encodeBpeToken(matcher.group(), out);
        }
    }

    private void encodeBpeToken(String token, List<Integer> out) {
        byte[] bytes = token.getBytes(StandardCharsets.UTF_8);
        List<String> symbols = new ArrayList<>(bytes.length);
        for (byte b : bytes) {
            symbols.add(String.valueOf(BYTE_TO_UNICODE[b & 0xff]));
        }
        List<String> merged = applyBpe(symbols);
        for (String item : merged) {
            out.add(vocab.getOrDefault(item, unknownTokenId));
        }
    }

    private List<String> applyBpe(List<String> symbols) {
        if (symbols.size() < 2) {
            return symbols;
        }
        List<String> work = new ArrayList<>(symbols);
        while (work.size() > 1) {
            int bestIndex = -1;
            int bestRank = Integer.MAX_VALUE;
            for (int i = 0; i < work.size() - 1; i++) {
                Integer rank = mergeRanks.get(pairKey(work.get(i), work.get(i + 1)));
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestIndex = i;
                }
            }
            if (bestIndex < 0) {
                break;
            }
            String merged = work.get(bestIndex) + work.get(bestIndex + 1);
            work.set(bestIndex, merged);
            work.remove(bestIndex + 1);
        }
        return work;
    }

    private List<Object> splitSpecialTokens(String text) {
        if (specialTokens.isEmpty()) {
            return List.of(text);
        }
        List<String> specials = new ArrayList<>(specialTokens.keySet());
        specials.sort(Comparator.comparingInt(String::length).reversed());
        List<Object> result = new ArrayList<>();
        int index = 0;
        while (index < text.length()) {
            String matched = null;
            for (String special : specials) {
                if (text.startsWith(special, index)) {
                    matched = special;
                    break;
                }
            }
            if (matched == null) {
                int next = nextSpecialIndex(text, specials, index + 1);
                result.add(text.substring(index, next));
                index = next;
            } else {
                result.add(specialTokens.get(matched));
                index += matched.length();
            }
        }
        return result;
    }

    private static int nextSpecialIndex(String text, List<String> specials, int from) {
        int best = text.length();
        for (String special : specials) {
            int found = text.indexOf(special, from);
            if (found >= 0 && found < best) {
                best = found;
            }
        }
        return best;
    }

    private static String decodeByteLevelText(String encoded) {
        byte[] bytes = new byte[encoded.length() * 4];
        int count = 0;
        for (int i = 0; i < encoded.length(); i++) {
            Byte b = UNICODE_TO_BYTE.get(encoded.charAt(i));
            if (b != null) {
                bytes[count++] = b;
            }
        }
        return new String(Arrays.copyOf(bytes, count), StandardCharsets.UTF_8);
    }

    private static Map<String, Integer> readVocab(Path vocabJson) throws IOException {
        JsonNode root = new ObjectMapper().readTree(vocabJson.toFile());
        if (!root.isObject()) {
            throw new IOException("CodeT5 vocab.json must be a JSON object: " + vocabJson);
        }
        Map<String, Integer> vocab = new LinkedHashMap<>();
        root.fields().forEachRemaining(entry -> vocab.put(entry.getKey(), entry.getValue().asInt()));
        return vocab;
    }

    private static String[] reverse(Map<String, Integer> vocab) throws IOException {
        int max = vocab.values().stream().mapToInt(Integer::intValue).max().orElseThrow();
        String[] out = new String[max + 1];
        Arrays.fill(out, "");
        for (Map.Entry<String, Integer> entry : vocab.entrySet()) {
            int id = entry.getValue();
            if (id < 0 || id >= out.length) {
                throw new IOException("Invalid CodeT5 vocab id for token " + entry.getKey() + ": " + id);
            }
            out[id] = entry.getKey();
        }
        return out;
    }

    private static Map<Long, Integer> readMerges(Path mergesTxt) throws IOException {
        Map<Long, Integer> mergeRanks = new LinkedHashMap<>();
        int rank = 0;
        for (String line : Files.readAllLines(mergesTxt, StandardCharsets.UTF_8)) {
            String trimmed = line.trim();
            if (trimmed.isEmpty() || trimmed.startsWith("#")) {
                continue;
            }
            int separator = trimmed.indexOf(' ');
            if (separator <= 0 || separator >= trimmed.length() - 1) {
                throw new IOException("Invalid CodeT5 merge entry: " + line);
            }
            String left = trimmed.substring(0, separator);
            String right = trimmed.substring(separator + 1);
            mergeRanks.put(pairKey(left, right), rank++);
        }
        return mergeRanks;
    }

    private static Map<String, Integer> readSpecialTokens(Path modelDir, Map<String, Integer> vocab) throws IOException {
        Map<String, Integer> specialTokens = new LinkedHashMap<>();
        putIfPresent(specialTokens, vocab, "<s>");
        putIfPresent(specialTokens, vocab, "</s>");
        putIfPresent(specialTokens, vocab, "<pad>");
        putIfPresent(specialTokens, vocab, "<unk>");
        putIfPresent(specialTokens, vocab, "<mask>");
        Path specialTokensMap = modelDir.resolve("special_tokens_map.json");
        if (!Files.isRegularFile(specialTokensMap)) {
            return specialTokens;
        }
        JsonNode root = new ObjectMapper().readTree(specialTokensMap.toFile());
        root.fields().forEachRemaining(entry -> addSpecialTokenValue(specialTokens, vocab, entry.getValue()));
        return specialTokens;
    }

    private static void addSpecialTokenValue(Map<String, Integer> specialTokens,
                                             Map<String, Integer> vocab,
                                             JsonNode value) {
        if (value == null) {
            return;
        }
        if (value.isTextual()) {
            putIfPresent(specialTokens, vocab, value.asText());
            return;
        }
        if (value.isObject() && value.has("content")) {
            putIfPresent(specialTokens, vocab, value.get("content").asText());
            return;
        }
        if (value.isArray()) {
            for (JsonNode child : value) {
                addSpecialTokenValue(specialTokens, vocab, child);
            }
        }
    }

    private static void putIfPresent(Map<String, Integer> specialTokens,
                                     Map<String, Integer> vocab,
                                     String token) {
        Integer id = vocab.get(token);
        if (id != null) {
            specialTokens.put(token, id);
        }
    }

    private static long pairKey(String left, String right) {
        return (((long) left.hashCode()) << 32) ^ (right.hashCode() & 0xffffffffL);
    }

    private static int[] toIntArray(List<Integer> values) {
        int[] out = new int[values.size()];
        for (int i = 0; i < values.size(); i++) {
            out[i] = values.get(i);
        }
        return out;
    }
}
