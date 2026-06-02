package com.aresstack.windirectml.inference.qwen;

import java.util.List;

/**
 * Stop-token policy for Qwen2.5-Coder-Instruct generation.
 *
 * <p>During auto-regressive generation, the decoder loop must stop
 * when any of the designated stop tokens is produced. For Qwen2.5-Coder
 * the stop tokens are:
 * <ul>
 *   <li>{@code <|endoftext|>} (ID 151643) – general end-of-text</li>
 *   <li>{@code <|im_end|>} (ID 151645) – ChatML turn-end marker</li>
 * </ul>
 *
 * <h2>Differences from Phi-3 stop policy</h2>
 * <ul>
 *   <li>Phi-3 stops on three tokens: {@code <|endoftext|>} (32000),
 *       {@code <|assistant|>} (32001), and {@code <|end|>} (32007).</li>
 *   <li>Qwen stops on two tokens: {@code <|endoftext|>} (151643) and
 *       {@code <|im_end|>} (151645). The ChatML {@code <|im_end|>}
 *       serves the same role as Phi-3's {@code <|end|>} token.</li>
 *   <li>Qwen does not need to stop on a role-specific token because
 *       ChatML uses a single generic end marker for all turns.</li>
 * </ul>
 *
 * <p><b>Runtime status:</b> Qwen model generation is <em>planned</em>.
 * This policy supports offline testing and preparation only.
 */
public final class QwenStopTokenPolicy {

    private QwenStopTokenPolicy() {
    } // utility class

    /**
     * End-of-text token: {@code <|endoftext|>}.
     */
    public static final int ENDOFTEXT_ID = QwenTokenizer.ENDOFTEXT_ID;

    /**
     * ChatML turn-end token: {@code <|im_end|>}.
     */
    public static final int IM_END_ID = QwenTokenizer.IM_END_ID;

    /**
     * All token IDs that terminate Qwen generation.
     */
    public static final List<Integer> STOP_TOKEN_IDS = List.of(ENDOFTEXT_ID, IM_END_ID);

    /**
     * Check whether a token ID should stop generation.
     *
     * @param tokenId the token ID produced by the decoder
     * @return true if generation should stop
     */
    public static boolean shouldStop(int tokenId) {
        return tokenId == ENDOFTEXT_ID || tokenId == IM_END_ID;
    }

    /**
     * Returns the list of stop token IDs for configuration/debugging.
     *
     * @return unmodifiable list of stop token IDs
     */
    public static List<Integer> stopTokenIds() {
        return STOP_TOKEN_IDS;
    }
}
