package com.aresstack.windirectml.inference.gemma;

import java.util.Arrays;

/**
 * Which token ids end generation (GEMMA-WARP-10b). Gemma instruct models end a turn with
 * {@code <end_of_turn>} (not always {@code <eos>}), so the policy holds a set of stop ids — typically
 * {@code <eos>} plus {@code <end_of_turn>} when its id is known (resolve via
 * {@code Gemma3Tokenizer.encode("<end_of_turn>", false)[0]}). A stop token ends generation and is
 * <b>not</b> part of the visible output.
 */
public final class Gemma3StopTokenPolicy {

    private final int[] stopIds;

    private Gemma3StopTokenPolicy(int[] stopIds) {
        this.stopIds = stopIds;
    }

    /** Stop on exactly these token ids. */
    public static Gemma3StopTokenPolicy of(int... stopIds) {
        int[] sorted = stopIds.clone();
        Arrays.sort(sorted);
        return new Gemma3StopTokenPolicy(sorted);
    }

    /** Stop on the config's {@code eos} token only. */
    public static Gemma3StopTokenPolicy ofEos(Gemma3Config config) {
        return of(config.eosTokenId());
    }

    /** Stop on {@code eos} and {@code <end_of_turn>}. */
    public static Gemma3StopTokenPolicy ofEosAndEndOfTurn(Gemma3Config config, int endOfTurnId) {
        return of(config.eosTokenId(), endOfTurnId);
    }

    public boolean isStop(int tokenId) {
        return Arrays.binarySearch(stopIds, tokenId) >= 0;
    }

    public int[] stopIds() {
        return stopIds.clone();
    }
}
