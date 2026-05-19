package com.aresstack.windirectml.encoder.tokenizer;

import com.aresstack.windirectml.encoder.EncoderTokenizer;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.Objects;

/**
 * WordPiece-Tokenizer für BERT-/MiniLM-Familien.
 * <p>
 * Liest das HuggingFace-{@code tokenizer.json}-Format und implementiert:
 * <ol>
 *   <li>Unicode-NFD-Normalisierung mit Strip-Accents.</li>
 *   <li>Lowercase (für {@code uncased}-Modelle wie MiniLM-L6-v2).</li>
 *   <li>Whitespace- und Punctuation-Pre-Tokenization.</li>
 *   <li>Greedy WordPiece-Subword-Segmentierung mit Continuing-Prefix {@code ##}.</li>
 *   <li>Spezial-Token-Wrapping mit {@code [CLS] ... [SEP]}.</li>
 *   <li>Truncation auf {@code maxSequenceLength}.</li>
 * </ol>
 * <p>
 * Decoder ist nicht implementiert (Encoder-Pfad braucht ihn nicht).
 */
public final class WordPieceTokenizer implements EncoderTokenizer {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final Map<String, Integer> vocab;
    private final String unkToken;
    private final String continuingPrefix;
    private final int maxInputCharsPerWord;
    private final boolean lowercase;
    private final boolean stripAccents;
    private final int clsId;
    private final int sepId;
    private final int padId;
    private final int unkId;
    private final int maxSequenceLength;

    private WordPieceTokenizer(Map<String, Integer> vocab,
                               String unkToken, String continuingPrefix,
                               int maxInputCharsPerWord, boolean lowercase, boolean stripAccents,
                               String clsToken, String sepToken, String padToken,
                               int maxSequenceLength) {
        this.vocab = vocab;
        this.unkToken = unkToken;
        this.continuingPrefix = continuingPrefix;
        this.maxInputCharsPerWord = maxInputCharsPerWord;
        this.lowercase = lowercase;
        this.stripAccents = stripAccents;
        this.clsId = required(vocab, clsToken);
        this.sepId = required(vocab, sepToken);
        this.padId = required(vocab, padToken);
        this.unkId = required(vocab, unkToken);
        this.maxSequenceLength = maxSequenceLength;
    }

    public static WordPieceTokenizer load(Path tokenizerJson) throws IOException {
        return load(tokenizerJson, 512);
    }

    public static WordPieceTokenizer load(Path tokenizerJson, int maxSequenceLength) throws IOException {
        JsonNode root = MAPPER.readTree(Files.newInputStream(tokenizerJson));

        // Normalizer (BertNormalizer)
        boolean lowercase = true;
        boolean stripAccents = true;
        JsonNode normalizer = root.get("normalizer");
        if (normalizer != null && !normalizer.isNull()) {
            if (normalizer.hasNonNull("lowercase"))    lowercase    = normalizer.get("lowercase").asBoolean();
            if (normalizer.hasNonNull("strip_accents")) stripAccents = normalizer.get("strip_accents").asBoolean();
        }

        // Model (WordPiece)
        JsonNode model = root.get("model");
        if (model == null || !"WordPiece".equals(textOrNull(model, "type"))) {
            throw new IOException("tokenizer.json model.type is not WordPiece");
        }
        String unkToken = textOr(model, "unk_token", "[UNK]");
        String continuingPrefix = textOr(model, "continuing_subword_prefix", "##");
        int maxInputCharsPerWord = model.hasNonNull("max_input_chars_per_word")
                ? model.get("max_input_chars_per_word").asInt(100) : 100;
        JsonNode vocabNode = model.get("vocab");
        if (vocabNode == null || !vocabNode.isObject()) {
            throw new IOException("tokenizer.json model.vocab missing");
        }
        Map<String, Integer> vocab = new HashMap<>(vocabNode.size() * 2);
        for (Iterator<Map.Entry<String, JsonNode>> it = vocabNode.fields(); it.hasNext(); ) {
            Map.Entry<String, JsonNode> e = it.next();
            vocab.put(e.getKey(), e.getValue().asInt());
        }

        // Special tokens (defaults match BERT)
        String clsToken = "[CLS]";
        String sepToken = "[SEP]";
        String padToken = "[PAD]";
        JsonNode added = root.get("added_tokens");
        if (added != null && added.isArray()) {
            for (JsonNode tok : added) {
                String content = textOrNull(tok, "content");
                if (content == null) continue;
                if (content.equalsIgnoreCase("[CLS]")) clsToken = content;
                else if (content.equalsIgnoreCase("[SEP]")) sepToken = content;
                else if (content.equalsIgnoreCase("[PAD]")) padToken = content;
            }
        }

        return new WordPieceTokenizer(vocab, unkToken, continuingPrefix,
                maxInputCharsPerWord, lowercase, stripAccents,
                clsToken, sepToken, padToken, maxSequenceLength);
    }

    // ── EncoderTokenizer API ─────────────────────────────────────────────

    @Override
    public Encoded encode(String text) {
        Objects.requireNonNull(text);
        List<String> preTokens = preTokenize(normalize(text));
        List<Integer> ids = new ArrayList<>();
        ids.add(clsId);
        for (String word : preTokens) {
            wordPiece(word, ids);
            if (ids.size() >= maxSequenceLength - 1) break;
        }
        ids.add(sepId);
        if (ids.size() > maxSequenceLength) {
            ids = new ArrayList<>(ids.subList(0, maxSequenceLength));
            ids.set(maxSequenceLength - 1, sepId);
        }
        int n = ids.size();
        int[] inputIds = new int[n];
        int[] mask = new int[n];
        int[] segments = new int[n];
        for (int i = 0; i < n; i++) {
            inputIds[i] = ids.get(i);
            mask[i] = 1;
            segments[i] = 0;
        }
        return new Encoded(inputIds, mask, segments);
    }

    /**
     * BERT NSP-style pair tokenization:
     * {@code [CLS] a [SEP] b [SEP]} with {@code token_type_ids = [0…0,1…1]}.
     * Both segments are truncated longest-first until the joint length
     * fits {@link #maxSequenceLength} (three special tokens reserved).
     */
    @Override
    public Encoded encodePair(String textA, String textB) {
        Objects.requireNonNull(textA, "textA");
        Objects.requireNonNull(textB, "textB");

        List<Integer> a = new ArrayList<>();
        for (String word : preTokenize(normalize(textA))) wordPiece(word, a);
        List<Integer> b = new ArrayList<>();
        for (String word : preTokenize(normalize(textB))) wordPiece(word, b);

        int reserved = 3; // [CLS] [SEP] [SEP]
        int budget = Math.max(0, maxSequenceLength - reserved);
        // Longest-first truncation – matches HuggingFace BertTokenizer's
        // default `truncation_strategy=longest_first`.
        while (a.size() + b.size() > budget) {
            if (a.size() >= b.size()) a.remove(a.size() - 1);
            else b.remove(b.size() - 1);
        }

        int n = 1 + a.size() + 1 + b.size() + 1;
        int[] inputIds = new int[n];
        int[] mask = new int[n];
        int[] segments = new int[n];
        int p = 0;
        inputIds[p] = clsId; segments[p] = 0; p++;
        for (int id : a) { inputIds[p] = id; segments[p] = 0; p++; }
        inputIds[p] = sepId; segments[p] = 0; p++;
        for (int id : b) { inputIds[p] = id; segments[p] = 1; p++; }
        inputIds[p] = sepId; segments[p] = 1; p++;
        for (int i = 0; i < n; i++) mask[i] = 1;
        return new Encoded(inputIds, mask, segments);
    }

    @Override public int padTokenId() { return padId; }
    @Override public int clsTokenId() { return clsId; }
    @Override public int sepTokenId() { return sepId; }
    @Override public int vocabSize() { return vocab.size(); }

    // ── Internals ────────────────────────────────────────────────────────

    private String normalize(String text) {
        String n = text;
        if (stripAccents) {
            n = Normalizer.normalize(n, Normalizer.Form.NFD);
            StringBuilder sb = new StringBuilder(n.length());
            for (int i = 0; i < n.length(); i++) {
                char c = n.charAt(i);
                if (Character.getType(c) != Character.NON_SPACING_MARK) sb.append(c);
            }
            n = sb.toString();
        }
        if (lowercase) n = n.toLowerCase();
        return n;
    }

    private static List<String> preTokenize(String text) {
        List<String> tokens = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (Character.isWhitespace(c) || isControl(c)) {
                flush(current, tokens);
            } else if (isPunctuation(c)) {
                flush(current, tokens);
                tokens.add(String.valueOf(c));
            } else {
                current.append(c);
            }
        }
        flush(current, tokens);
        return tokens;
    }

    private static void flush(StringBuilder sb, List<String> sink) {
        if (sb.length() > 0) {
            sink.add(sb.toString());
            sb.setLength(0);
        }
    }

    private static boolean isControl(char c) {
        if (c == '\t' || c == '\n' || c == '\r') return false;
        int type = Character.getType(c);
        return type == Character.CONTROL || type == Character.FORMAT;
    }

    private static boolean isPunctuation(char c) {
        if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64)
                || (c >= 91 && c <= 96) || (c >= 123 && c <= 126)) return true;
        int type = Character.getType(c);
        return type >= Character.DASH_PUNCTUATION && type <= Character.OTHER_PUNCTUATION;
    }

    private void wordPiece(String word, List<Integer> out) {
        if (word.length() > maxInputCharsPerWord) {
            out.add(unkId);
            return;
        }
        int start = 0;
        List<Integer> sub = new ArrayList<>();
        while (start < word.length()) {
            int end = word.length();
            Integer match = null;
            while (start < end) {
                String piece = word.substring(start, end);
                if (start > 0) piece = continuingPrefix + piece;
                Integer id = vocab.get(piece);
                if (id != null) { match = id; break; }
                end--;
            }
            if (match == null) {
                out.add(unkId);
                return;
            }
            sub.add(match);
            start = end;
        }
        out.addAll(sub);
    }

    private static String textOrNull(JsonNode node, String field) {
        JsonNode v = node.get(field);
        return v != null && v.isTextual() ? v.asText() : null;
    }

    private static String textOr(JsonNode node, String field, String fallback) {
        String s = textOrNull(node, field);
        return s != null ? s : fallback;
    }

    private static int required(Map<String, Integer> vocab, String tokenLiteral) {
        Integer id = vocab.get(tokenLiteral);
        if (id == null) throw new IllegalArgumentException("vocab missing required token: " + tokenLiteral);
        return id;
    }
}

