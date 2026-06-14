package com.aresstack.windirectml.inference.gemma;

/**
 * Gemma 3 instruction chat template. A single user turn followed by the model turn opener:
 * {@code <start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n}. The tokenizer prepends
 * {@code <bos>} and recognizes {@code <start_of_turn>}/{@code <end_of_turn>} as special tokens.
 */
public final class Gemma3ChatTemplate {

    public static final String START_OF_TURN = "<start_of_turn>";
    public static final String END_OF_TURN = "<end_of_turn>";

    private Gemma3ChatTemplate() {
    }

    /** Render a single user turn with the model-turn opener (no BOS; the tokenizer adds it). */
    public static String renderUserTurn(String userContent) {
        String content = userContent == null ? "" : userContent;
        return START_OF_TURN + "user\n" + content + END_OF_TURN + "\n" + START_OF_TURN + "model\n";
    }
}
