package com.aresstack.windirectml.inference.phi3;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.*;

/**
 * BPE tokenizer for Phi-3-mini, loaded from HuggingFace {@code tokenizer.json}.
 *
 * <p>Encoding pipeline:
 * <ol>
 *   <li>Normalize: prepend {@code \u2581}, replace all spaces with {@code \u2581}</li>
 *   <li>Split into unicode characters</li>
 *   <li>Byte fallback for unknown characters (UTF-8 bytes as {@code <0xHH>} tokens)</li>
 *   <li>Apply BPE merges in priority order</li>
 *   <li>Map merged tokens to IDs via vocab</li>
 * </ol>
 *
 * <p>Decoding pipeline:
 * <ol>
 *   <li>Map IDs to token strings</li>
 *   <li>Concatenate</li>
 *   <li>Decode byte fallback tokens</li>
 *   <li>Replace {@code \u2581} with space</li>
 *   <li>Strip leading space</li>
 * </ol>
 */
public final class Phi3Tokenizer {

    private static final Logger log = LoggerFactory.getLogger(Phi3Tokenizer.class);

    /**
     * SentencePiece space marker.
     */
    private static final char SP = '\u2581';

    // ── Special token IDs ────────────────────────────────────────────────

    public static final int UNK_ID = 0;       // <unk>
    public static final int BOS_ID = 1;       // <s>
    public static final int EOS_ID = 2;       // </s>
    public static final int ENDOFTEXT_ID = 32000; // <|endoftext|>
    public static final int ASSISTANT_ID = 32001; // <|assistant|>
    public static final int SYSTEM_ID = 32006;    // <|system|>
    public static final int END_ID = 32007;       // <|end|>
    public static final int USER_ID = 32010;      // <|user|>

    /**
     * EOS token IDs that terminate generation.
     */
    public static final int[] EOS_IDS = {ENDOFTEXT_ID, ASSISTANT_ID, END_ID};

    // ── Internal state ───────────────────────────────────────────────────

    private final Map<String, Integer> vocab;        // token string -> id
    private final String[] idToToken;                // id -> token string
    private final Map<Long, Integer> mergeRanks;     // packed pair hash -> rank
    private final String[][] mergeList;              // ordered merge pairs for debug
    private final Map<String, Integer> specialTokens;// special token -> id
    private final int vocabSize;

    private Phi3Tokenizer(Map<String, Integer> vocab, String[] idToToken,
                          Map<Long, Integer> mergeRanks, String[][] mergeList,
                          Map<String, Integer> specialTokens) {
        this.vocab = vocab;
        this.idToToken = idToToken;
        this.mergeRanks = mergeRanks;
        this.mergeList = mergeList;
        this.specialTokens = specialTokens;
        this.vocabSize = idToToken.length;
    }

    // ── Factory ──────────────────────────────────────────────────────────

    /**
     * Load tokenizer from HuggingFace {@code tokenizer.json}.
     */
    public static Phi3Tokenizer load(Path tokenizerJson) throws IOException {
        log.info("Loading tokenizer from {}", tokenizerJson);
        ObjectMapper om = new ObjectMapper();
        JsonNode root = om.readTree(tokenizerJson.toFile());
        JsonNode model = root.get("model");

        // --- Vocab ---
        JsonNode vocabNode = model.get("vocab");
        Map<String, Integer> vocab = new LinkedHashMap<>(40000);
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
                specialTokens.put(content, id);
                vocab.put(content, id);
                if (id > maxId) maxId = id;
            }
        }

        // --- Build reverse mapping ---
        String[] idToToken = new String[maxId + 1];
        Arrays.fill(idToToken, "<unk>");
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
        String[][] mergeList = new String[mergeCount][2];
        for (int i = 0; i < mergeCount; i++) {
            String merge = mergesNode.get(i).asText();
            int sep = merge.indexOf(' ');
            String a = merge.substring(0, sep);
            String b = merge.substring(sep + 1);
            mergeList[i] = new String[]{a, b};
            mergeRanks.put(pairKey(a, b), i);
        }

        log.info("Loaded tokenizer: {} vocab, {} merges, {} special tokens",
                vocab.size(), mergeCount, specialTokens.size());
        return new Phi3Tokenizer(vocab, idToToken, mergeRanks, mergeList, specialTokens);
    }

    // ── Encode ───────────────────────────────────────────────────────────

    /**
     * Encode a text string to token IDs.
     * Does NOT add BOS/EOS tokens.
     */
    public int[] encode(String text) {
        if (text == null || text.isEmpty()) return new int[0];

        // Check for special tokens first and split around them
        List<Object> segments = splitSpecialTokens(text);

        List<Integer> ids = new ArrayList<>();
        for (Object seg : segments) {
            if (seg instanceof Integer specialId) {
                ids.add(specialId);
            } else {
                String s = (String) seg;
                encodeSegment(s, ids);
            }
        }

        return ids.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Encode a regular text segment (no special tokens) using BPE.
     */
    private void encodeSegment(String text, List<Integer> ids) {
        // Step 1: Normalize - prepend SP, replace spaces with SP
        String normalized = SP + text.replace(' ', SP);

        // Step 2: Split into initial symbols (chars or byte fallback)
        List<String> symbols = new ArrayList<>();
        for (int i = 0; i < normalized.length(); ) {
            int cp = normalized.codePointAt(i);
            String ch = new String(Character.toChars(cp));
            if (vocab.containsKey(ch)) {
                symbols.add(ch);
            } else {
                // Byte fallback: encode UTF-8 bytes
                byte[] utf8 = ch.getBytes(StandardCharsets.UTF_8);
                for (byte b : utf8) {
                    symbols.add(String.format("<0x%02X>", b & 0xFF));
                }
            }
            i += Character.charCount(cp);
        }

        // Step 3: BPE merge loop
        while (symbols.size() > 1) {
            // Find the highest-priority merge (lowest rank)
            int bestIdx = -1;
            int bestRank = Integer.MAX_VALUE;
            for (int i = 0; i < symbols.size() - 1; i++) {
                Integer rank = mergeRanks.get(pairKey(symbols.get(i), symbols.get(i + 1)));
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestIdx = i;
                }
            }
            if (bestIdx < 0) break; // No more merges possible

            // Apply the merge
            String merged = symbols.get(bestIdx) + symbols.get(bestIdx + 1);
            symbols.set(bestIdx, merged);
            symbols.remove(bestIdx + 1);
        }

        // Step 4: Map symbols to IDs
        for (String sym : symbols) {
            Integer id = vocab.get(sym);
            ids.add(id != null ? id : UNK_ID);
        }
    }

    /**
     * Split text around special tokens, returning a list of alternating
     * String segments and Integer special-token IDs.
     */
    private List<Object> splitSpecialTokens(String text) {
        List<Object> result = new ArrayList<>();
        // Sort special tokens by length descending (greedy match)
        List<Map.Entry<String, Integer>> sorted = new ArrayList<>(specialTokens.entrySet());
        sorted.sort((a, b) -> Integer.compare(b.getKey().length(), a.getKey().length()));

        splitRecursive(text, sorted, 0, result);
        return result;
    }

    private void splitRecursive(String text, List<Map.Entry<String, Integer>> tokens,
                                int depth, List<Object> result) {
        if (text.isEmpty()) return;

        // Find earliest special token
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

        // Text before the special token
        if (bestPos > 0) {
            result.add(text.substring(0, bestPos));
        }
        // The special token itself
        result.add(bestToken.getValue());
        // Text after
        String after = text.substring(bestPos + bestToken.getKey().length());
        if (!after.isEmpty()) {
            splitRecursive(after, tokens, depth + 1, result);
        }
    }

    // ── Decode ───────────────────────────────────────────────────────────

    /**
     * Decode token IDs back to text.
     */
    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder();
        List<Byte> pendingBytes = new ArrayList<>();

        for (int id : ids) {
            String token = (id >= 0 && id < idToToken.length) ? idToToken[id] : "<unk>";

            // Handle byte fallback tokens
            if (token.startsWith("<0x") && token.endsWith(">") && token.length() == 6) {
                int b = Integer.parseInt(token.substring(3, 5), 16);
                pendingBytes.add((byte) b);
                continue;
            }

            // Flush pending bytes
            if (!pendingBytes.isEmpty()) {
                byte[] bytes = new byte[pendingBytes.size()];
                for (int i = 0; i < bytes.length; i++) bytes[i] = pendingBytes.get(i);
                sb.append(new String(bytes, StandardCharsets.UTF_8));
                pendingBytes.clear();
            }

            sb.append(token);
        }

        // Flush remaining bytes
        if (!pendingBytes.isEmpty()) {
            byte[] bytes = new byte[pendingBytes.size()];
            for (int i = 0; i < bytes.length; i++) bytes[i] = pendingBytes.get(i);
            sb.append(new String(bytes, StandardCharsets.UTF_8));
        }

        // Replace SP with space, strip leading space
        String result = sb.toString().replace(SP, ' ');
        if (result.startsWith(" ")) result = result.substring(1);
        return result;
    }

    /**
     * Decode a single token ID to its string representation (raw, no SP replacement).
     */
    public String decodeToken(int id) {
        return (id >= 0 && id < idToToken.length) ? idToToken[id] : "<unk>";
    }

    // ── Chat template ────────────────────────────────────────────────────

    /**
     * A single message in a multi-turn conversation.
     *
     * @param role    either {@code "user"} or {@code "assistant"}
     * @param content the message text
     */
    public record ChatMessage(String role, String content) {
        public static ChatMessage user(String content) {
            return new ChatMessage("user", content);
        }

        public static ChatMessage assistant(String content) {
            return new ChatMessage("assistant", content);
        }
    }

    /**
     * Format a single-turn chat prompt using Phi-3 chat template.
     *
     * <pre>
     * &lt;|system|&gt;
     * {systemPrompt}&lt;|end|&gt;
     * &lt;|user|&gt;
     * {userMessage}&lt;|end|&gt;
     * &lt;|assistant|&gt;
     * </pre>
     *
     * @return formatted prompt string ready for encoding
     */
    public String formatChat(String systemPrompt, String userMessage) {
        return formatMultiTurnChat(systemPrompt, List.of(ChatMessage.user(userMessage)));
    }

    /**
     * Format a multi-turn chat prompt using Phi-3 chat template.
     *
     * <pre>
     * &lt;|system|&gt;
     * {systemPrompt}&lt;|end|&gt;
     * &lt;|user|&gt;
     * {message1}&lt;|end|&gt;
     * &lt;|assistant|&gt;
     * {reply1}&lt;|end|&gt;
     * &lt;|user|&gt;
     * {message2}&lt;|end|&gt;
     * &lt;|assistant|&gt;
     * </pre>
     *
     * @param systemPrompt optional system prompt (may be {@code null})
     * @param messages     ordered list of user/assistant messages; the last
     *                     message should typically be a user message
     * @return formatted prompt string ready for encoding
     */
    public String formatMultiTurnChat(String systemPrompt, List<ChatMessage> messages) {
        StringBuilder sb = new StringBuilder();
        if (systemPrompt != null && !systemPrompt.isEmpty()) {
            sb.append("<|system|>\n").append(systemPrompt).append("<|end|>\n");
        }
        for (ChatMessage msg : messages) {
            switch (msg.role()) {
                case "user" -> sb.append("<|user|>\n").append(msg.content()).append("<|end|>\n");
                case "assistant" -> sb.append("<|assistant|>\n").append(msg.content()).append("<|end|>\n");
            }
        }
        sb.append("<|assistant|>\n");
        return sb.toString();
    }

    // ── Utility ──────────────────────────────────────────────────────────

    public int vocabSize() {
        return vocabSize;
    }

    /**
     * Returns an unmodifiable view of the special tokens map (for debugging).
     */
    public Map<String, Integer> specialTokensMap() {
        return Collections.unmodifiableMap(specialTokens);
    }

    public boolean isEos(int tokenId) {
        for (int eos : EOS_IDS) {
            if (tokenId == eos) return true;
        }
        return false;
    }

    /**
     * Hash key for a merge pair. Uses a simple hash combining both strings.
     */
    private static long pairKey(String a, String b) {
        return ((long) a.hashCode()) << 32 | (b.hashCode() & 0xFFFFFFFFL);
    }
}
