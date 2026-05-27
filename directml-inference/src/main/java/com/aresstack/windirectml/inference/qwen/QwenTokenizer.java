package com.aresstack.windirectml.inference.qwen;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * BPE tokenizer for Qwen2.5-Coder, loaded from HuggingFace {@code tokenizer.json}.
 *
 * <h2>Differences from Phi-3 tokenizer ({@link com.aresstack.windirectml.inference.phi3.Phi3Tokenizer})</h2>
 * <ul>
 *   <li><b>Pre-tokenization:</b> Qwen uses a regex-based pre-tokenizer that splits on
 *       whitespace boundaries and punctuation (similar to GPT-4 pattern), whereas Phi-3
 *       uses SentencePiece-style ▁ (U+2581) space markers.</li>
 *   <li><b>No SentencePiece space normalization:</b> Qwen BPE operates directly on
 *       UTF-8 byte sequences without the leading ▁ prefix that Phi-3 requires.</li>
 *   <li><b>Byte-level BPE:</b> Qwen maps raw bytes to single-character vocab entries
 *       (byte-to-unicode mapping like GPT-2/GPT-4), while Phi-3 uses explicit
 *       {@code <0xHH>} fallback tokens.</li>
 *   <li><b>Vocabulary size:</b> Qwen2.5-Coder uses ~152k tokens vs Phi-3's ~32k.</li>
 *   <li><b>Chat template:</b> Qwen uses ChatML ({@code <|im_start|>/<|im_end|>}),
 *       while Phi-3 uses {@code <|system|>/<|user|>/<|assistant|>/<|end|>}.</li>
 *   <li><b>Stop tokens:</b> Qwen stops on {@code <|endoftext|>} and {@code <|im_end|>},
 *       while Phi-3 stops on {@code <|endoftext|>}, {@code <|assistant|>}, and
 *       {@code <|end|>}.</li>
 * </ul>
 *
 * <p>Encoding pipeline:
 * <ol>
 *   <li>Pre-tokenize: split input using regex pattern (whitespace-aware word boundaries)</li>
 *   <li>Encode each pre-token into bytes, map bytes to base vocab characters</li>
 *   <li>Apply BPE merges in priority order</li>
 *   <li>Map merged tokens to IDs via vocab</li>
 * </ol>
 *
 * <p>Decoding pipeline:
 * <ol>
 *   <li>Map IDs to token strings</li>
 *   <li>Concatenate</li>
 *   <li>Reverse byte-to-unicode mapping to recover raw bytes</li>
 *   <li>Decode UTF-8 bytes to string</li>
 * </ol>
 *
 * <p><b>Runtime status:</b> Qwen model generation is <em>planned</em>.
 * This tokenizer supports offline prompt preparation and testing only.
 */
public final class QwenTokenizer {

    private static final Logger log = LoggerFactory.getLogger(QwenTokenizer.class);

    // ── GPT-2/Qwen byte-to-unicode mapping ───────────────────────────────

    private static final char[] BYTE_TO_UNICODE = new char[256];
    private static final Map<Character, Byte> UNICODE_TO_BYTE = new HashMap<>(512);

    static {
        // Build the byte ↔ unicode mapping used by GPT-2/Qwen byte-level BPE.
        // Printable bytes map to themselves; others get shifted to high Unicode.
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

    // ── Pre-tokenization regex ───────────────────────────────────────────
    // Matches the pattern used by Qwen/GPT-4 tokenizers.
    private static final Pattern PRE_TOKENIZE_PATTERN = Pattern.compile(
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    );

    // ── Special token IDs (Qwen2.5-Coder-Instruct) ───────────────────────

    /** End-of-text token ID. */
    public static final int ENDOFTEXT_ID = 151643;

    /** ChatML start marker: {@code <|im_start|>}. */
    public static final int IM_START_ID = 151644;

    /** ChatML end marker: {@code <|im_end|>}. */
    public static final int IM_END_ID = 151645;

    // ── Internal state ───────────────────────────────────────────────────

    private final Map<String, Integer> vocab;        // token string -> id
    private final String[] idToToken;                // id -> token string
    private final Map<Long, Integer> mergeRanks;     // packed pair hash -> rank
    private final Map<String, Integer> specialTokens;// special token -> id
    private final int vocabSize;

    private QwenTokenizer(Map<String, Integer> vocab, String[] idToToken,
                          Map<Long, Integer> mergeRanks,
                          Map<String, Integer> specialTokens) {
        this.vocab = vocab;
        this.idToToken = idToToken;
        this.mergeRanks = mergeRanks;
        this.specialTokens = specialTokens;
        this.vocabSize = idToToken.length;
    }

    // ── Factory ──────────────────────────────────────────────────────────

    /**
     * Load tokenizer from HuggingFace {@code tokenizer.json} (Qwen2.5-Coder format).
     *
     * <p>The file must contain:
     * <ul>
     *   <li>{@code model.vocab} – the BPE vocabulary mapping</li>
     *   <li>{@code model.merges} – ordered BPE merge rules</li>
     *   <li>{@code added_tokens} – special tokens with their IDs</li>
     * </ul>
     *
     * @param tokenizerJson path to the tokenizer.json file
     * @return loaded tokenizer instance
     * @throws IOException if the file cannot be read or parsed
     */
    public static QwenTokenizer load(Path tokenizerJson) throws IOException {
        log.info("Loading Qwen tokenizer from {}", tokenizerJson);
        ObjectMapper om = new ObjectMapper();
        JsonNode root = om.readTree(tokenizerJson.toFile());
        JsonNode model = root.get("model");

        // --- Vocab ---
        JsonNode vocabNode = model.get("vocab");
        Map<String, Integer> vocab = new LinkedHashMap<>(160000);
        Iterator<Map.Entry<String, JsonNode>> it = vocabNode.fields();
        while (it.hasNext()) {
            Map.Entry<String, JsonNode> e = it.next();
            vocab.put(e.getKey(), e.getValue().asInt());
        }

        // --- Added tokens (special tokens mapped by content) ---
        Map<String, Integer> specialTokens = new LinkedHashMap<>();
        JsonNode addedTokens = root.get("added_tokens");
        int maxId = vocab.values().stream().mapToInt(Integer::intValue).max().orElse(0);
        if (addedTokens != null && addedTokens.isArray()) {
            for (JsonNode at : addedTokens) {
                String content = at.get("content").asText();
                int id = at.get("id").asInt();
                boolean special = at.has("special") && at.get("special").asBoolean();
                if (special) {
                    specialTokens.put(content, id);
                }
                vocab.put(content, id);
                if (id > maxId) maxId = id;
            }
        }

        // --- Build reverse mapping ---
        String[] idToToken = new String[maxId + 1];
        Arrays.fill(idToToken, "");
        for (Map.Entry<String, Integer> e : vocab.entrySet()) {
            int id = e.getValue();
            if (id >= 0 && id < idToToken.length) {
                idToToken[id] = e.getKey();
            }
        }

        // --- Merges ---
        JsonNode mergesNode = model.get("merges");
        int mergeCount = mergesNode.size();
        Map<Long, Integer> mergeRanks = new HashMap<>(mergeCount * 2);
        for (int i = 0; i < mergeCount; i++) {
            String merge = mergesNode.get(i).asText();
            int sep = merge.indexOf(' ');
            String a = merge.substring(0, sep);
            String b = merge.substring(sep + 1);
            mergeRanks.put(pairKey(a, b), i);
        }

        log.info("Loaded Qwen tokenizer: {} vocab, {} merges, {} special tokens",
                vocab.size(), mergeCount, specialTokens.size());
        return new QwenTokenizer(vocab, idToToken, mergeRanks, specialTokens);
    }

    // ── Encode ───────────────────────────────────────────────────────────

    /**
     * Encode a text string to token IDs.
     * Does NOT add BOS/EOS tokens automatically.
     *
     * @param text input text to encode
     * @return array of token IDs
     */
    public int[] encode(String text) {
        if (text == null || text.isEmpty()) return new int[0];

        // Split around special tokens first
        List<Object> segments = splitSpecialTokens(text);

        List<Integer> ids = new ArrayList<>();
        for (Object seg : segments) {
            if (seg instanceof Integer specialId) {
                ids.add(specialId);
            } else {
                String s = (String) seg;
                encodeText(s, ids);
            }
        }

        return ids.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Encode a regular text segment (no special tokens) using byte-level BPE.
     */
    private void encodeText(String text, List<Integer> ids) {
        // Pre-tokenize using regex pattern
        Matcher matcher = PRE_TOKENIZE_PATTERN.matcher(text);
        while (matcher.find()) {
            String preToken = matcher.group();
            encodeBpeToken(preToken, ids);
        }
    }

    /**
     * Apply byte-level BPE to a single pre-token.
     */
    private void encodeBpeToken(String preToken, List<Integer> ids) {
        // Convert to byte-level unicode representation
        byte[] bytes = preToken.getBytes(StandardCharsets.UTF_8);
        List<String> symbols = new ArrayList<>(bytes.length);
        for (byte b : bytes) {
            symbols.add(String.valueOf(BYTE_TO_UNICODE[b & 0xFF]));
        }

        // BPE merge loop
        while (symbols.size() > 1) {
            int bestIdx = -1;
            int bestRank = Integer.MAX_VALUE;
            for (int i = 0; i < symbols.size() - 1; i++) {
                Integer rank = mergeRanks.get(pairKey(symbols.get(i), symbols.get(i + 1)));
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestIdx = i;
                }
            }
            if (bestIdx < 0) break;

            String merged = symbols.get(bestIdx) + symbols.get(bestIdx + 1);
            symbols.set(bestIdx, merged);
            symbols.remove(bestIdx + 1);
        }

        // Map symbols to IDs
        for (String sym : symbols) {
            Integer id = vocab.get(sym);
            if (id != null) {
                ids.add(id);
            } else {
                // Unknown token – encode individual bytes
                for (char c : sym.toCharArray()) {
                    Integer byteId = vocab.get(String.valueOf(c));
                    ids.add(byteId != null ? byteId : 0);
                }
            }
        }
    }

    /**
     * Split text around special tokens, returning a list of alternating
     * String segments and Integer special-token IDs.
     */
    private List<Object> splitSpecialTokens(String text) {
        List<Object> result = new ArrayList<>();
        List<Map.Entry<String, Integer>> sorted = new ArrayList<>(specialTokens.entrySet());
        sorted.sort((a, b) -> Integer.compare(b.getKey().length(), a.getKey().length()));
        splitRecursive(text, sorted, result);
        return result;
    }

    private void splitRecursive(String text, List<Map.Entry<String, Integer>> tokens,
                                List<Object> result) {
        if (text.isEmpty()) return;

        int bestPos = text.length();
        Map.Entry<String, Integer> bestToken = null;
        for (Map.Entry<String, Integer> entry : tokens) {
            int pos = text.indexOf(entry.getKey());
            if (pos >= 0 && pos < bestPos) {
                bestPos = pos;
                bestToken = entry;
            }
        }

        if (bestToken == null) {
            result.add(text);
            return;
        }

        if (bestPos > 0) {
            result.add(text.substring(0, bestPos));
        }
        result.add(bestToken.getValue());
        String after = text.substring(bestPos + bestToken.getKey().length());
        if (!after.isEmpty()) {
            splitRecursive(after, tokens, result);
        }
    }

    // ── Decode ───────────────────────────────────────────────────────────

    /**
     * Decode token IDs back to text.
     *
     * @param ids array of token IDs
     * @return decoded text string
     */
    public String decode(int[] ids) {
        StringBuilder tokenChars = new StringBuilder();

        for (int id : ids) {
            if (id < 0 || id >= idToToken.length) continue;
            String token = idToToken[id];
            // Skip special tokens in decode output
            if (specialTokens.containsKey(token)) continue;
            tokenChars.append(token);
        }

        // Reverse byte-to-unicode mapping
        char[] chars = tokenChars.toString().toCharArray();
        byte[] bytes = new byte[chars.length];
        int byteCount = 0;
        for (char c : chars) {
            Byte b = UNICODE_TO_BYTE.get(c);
            if (b != null) {
                bytes[byteCount++] = b;
            }
            // Characters not in the reverse map are dropped (shouldn't happen
            // for well-formed token sequences)
        }

        return new String(bytes, 0, byteCount, StandardCharsets.UTF_8);
    }

    /**
     * Decode a single token ID to its raw token string (no byte-level reversal).
     *
     * @param id token ID
     * @return raw token string as stored in vocabulary
     */
    public String decodeToken(int id) {
        return (id >= 0 && id < idToToken.length) ? idToToken[id] : "";
    }

    // ── Utility ──────────────────────────────────────────────────────────

    /** Returns the vocabulary size. */
    public int vocabSize() { return vocabSize; }

    /** Returns an unmodifiable view of the special tokens map (for debugging). */
    public Map<String, Integer> specialTokensMap() {
        return Collections.unmodifiableMap(specialTokens);
    }

    /**
     * Check whether a token ID is an end-of-sequence token.
     *
     * @param tokenId the token ID to check
     * @return true if this token should terminate generation
     */
    public boolean isEos(int tokenId) {
        return tokenId == ENDOFTEXT_ID || tokenId == IM_END_ID;
    }

    /**
     * Hash key for a merge pair.
     */
    private static long pairKey(String a, String b) {
        return ((long) a.hashCode()) << 32 | (b.hashCode() & 0xFFFFFFFFL);
    }
}
