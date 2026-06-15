package com.aresstack.windirectml.inference.gemma;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Formats a {@link Gemma3NativeWarpProfile} into the human-readable "Gemma native WARP profile" block for
 * the Workbench (GEMMA-WORKBENCH-PROFILING-1). Swing-free so it can be unit-tested directly.
 *
 * <p>{@link #detailed} mirrors the spec layout (runtime mode / backend / output mode / package /
 * tokenizer / prompt template / token counts, then a {@code load:} group, a {@code generation:} group and
 * a {@code WARP counters:} group). {@link #summary} is the short block shown when the profile toggle is
 * off — enough to see the runtime mode, token counts and total without the per-phase detail.</p>
 */
public final class Gemma3NativeWarpProfileReport {

    private Gemma3NativeWarpProfileReport() {
    }

    /**
     * The detailed profile block.
     *
     * @param p                  the runtime-measured phase/counter profile
     * @param runtimeMode        e.g. {@code native-warp-experimental}
     * @param backend            e.g. {@code WARP}
     * @param outputMode         {@code streaming} or {@code buffered}
     * @param packageName        the runtime package file name (e.g. {@code model_gemma3.wdmlpack})
     * @param tokenizerName      the tokenizer file name (e.g. {@code tokenizer.json})
     * @param promptTemplate     the active prompt-template label (e.g. {@code NONE}, {@code summarize})
     * @param effectivePromptChars length of the rendered prompt actually fed to the tokenizer
     * @param promptTemplateMs   panel-measured prompt-template render time (ms)
     * @param grandTotalMs       panel-measured end-to-end time ("loaded and generated in"), ms
     */
    public static List<String> detailed(Gemma3NativeWarpProfile p, String runtimeMode, String backend,
                                        String outputMode, String packageName, String tokenizerName,
                                        String promptTemplate, int effectivePromptChars,
                                        long promptTemplateMs, long grandTotalMs) {
        Objects.requireNonNull(p, "profile");
        List<String> out = new ArrayList<>();
        out.add("Gemma native WARP profile:");
        out.add("  runtime mode: " + runtimeMode);
        out.add("  backend: " + backend);
        out.add("  output mode: " + outputMode);
        out.add("  package: " + packageName);
        out.add("  tokenizer: " + tokenizerName);
        out.add("  prompt template: " + promptTemplate);
        out.add("  effective prompt chars: " + effectivePromptChars);
        out.add("  prompt tokens: " + p.promptTokens());
        out.add("  output tokens: " + p.outputTokens());
        out.add("");
        out.add("  load:");
        out.add("    package open: " + ms(p.packageOpenMs()));
        out.add("    tokenizer load: " + ms(p.tokenizerLoadMs()));
        out.add("    weight load: " + ms(p.weightLoadMs()));
        out.add("    WARP/session init: " + ms(p.sessionInitMs()));
        out.add("");
        out.add("  generation:");
        out.add("    tokenize: " + ms(p.tokenizeMs()));
        out.add("    prompt template: " + ms(promptTemplateMs));
        out.add("    prefill: " + ms(p.prefillMs()));
        out.add("    decode total: " + ms(p.decodeTotalMs()));
        out.add("    decode avg/token: " + ms2(p.decodeAvgPerTokenMs()));
        // (decode avg/token is a duration; the per-token counters below are counts, not ms)
        out.add("    detokenize: " + ms(p.detokenizeMs()));
        out.add("    total: " + ms(grandTotalMs));
        out.add("");
        out.add("  WARP counters:");
        out.add("    submits: " + p.submits());
        out.add("    fence waits: " + p.fenceWaits());
        out.add("    readbacks: " + p.readbacks());
        out.add("    submits/token: " + num2(p.submitsPerToken()));
        out.add("    fence waits/token: " + num2(p.fenceWaitsPerToken()));
        out.add("    readbacks/token: " + num2(p.readbacksPerToken()));
        return out;
    }

    /** The short block shown when the profile toggle is off. */
    public static List<String> summary(String runtimeMode, int promptTokens, int outputTokens,
                                       String finishReason, long grandTotalMs) {
        List<String> out = new ArrayList<>();
        out.add("Model loaded and generated in " + grandTotalMs + " ms");
        out.add("Runtime mode: " + runtimeMode);
        out.add("Prompt tokens: " + promptTokens);
        out.add("Output tokens: " + outputTokens);
        out.add("Finish reason: " + finishReason);
        return out;
    }

    private static String ms(long v) {
        return v + " ms";
    }

    private static String ms2(double v) {
        return String.format(Locale.ROOT, "%.2f ms", v);
    }

    private static String num2(double v) {
        return String.format(Locale.ROOT, "%.2f", v);
    }
}
