package com.aresstack.windirectml.inference.prompt;

import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Resolves a model id to its {@link PromptStrategy}.
 *
 * <p>This is the single source of truth that ties a concrete model to (a) the
 * tasks it can serve and (b) how its prompt is rendered. Both the inference
 * engines (to format prompts) and the Workbench UI (to offer task options) call
 * into this factory, so the two can never drift apart.</p>
 */
public final class PromptStrategies {

    private PromptStrategies() {
    }

    /**
     * Returns the strategy for {@code modelId}. Never {@code null}; unknown ids
     * fall back to {@link RawPromptStrategy}.
     */
    public static PromptStrategy forModel(String modelId) {
        String id = Objects.toString(modelId, "").toLowerCase(Locale.ROOT);

        if (id.contains("gemma-3") || id.contains("gemma3")) {
            return new Gemma3PromptStrategy(ChatTaskInstructions.standard(), EnumSet.of(
                    PromptTask.NONE,
                    PromptTask.SUMMARIZE,
                    PromptTask.TRANSLATE_TO_GERMAN,
                    PromptTask.TRANSLATE_TO_ENGLISH,
                    PromptTask.EXPLAIN_CODE));
        }
        if (id.contains("smollm2")) {
            return chatMl(EnumSet.of(
                    PromptTask.NONE,
                    PromptTask.SUMMARIZE,
                    PromptTask.TRANSLATE_TO_GERMAN,
                    PromptTask.TRANSLATE_TO_ENGLISH));
        }
        if (id.contains("qwen2.5-coder") || id.contains("qwen2.5") || id.contains("qwen")) {
            return chatMl(EnumSet.of(
                    PromptTask.NONE,
                    PromptTask.SUMMARIZE,
                    PromptTask.TRANSLATE_TO_GERMAN,
                    PromptTask.TRANSLATE_TO_ENGLISH,
                    PromptTask.EXPLAIN_CODE));
        }
        if (id.contains("phi-3") || id.contains("phi3")) {
            return new Phi3PromptStrategy(ChatTaskInstructions.standard(), EnumSet.of(
                    PromptTask.NONE,
                    PromptTask.SUMMARIZE,
                    PromptTask.TRANSLATE_TO_GERMAN,
                    PromptTask.TRANSLATE_TO_ENGLISH));
        }
        if (id.contains("codet5-base-multi-sum")) {
            return new T5PromptStrategy(Map.of(PromptTask.SUMMARIZE, "summarize"));
        }
        if (id.contains("codet5-small") || id.contains("codet5")) {
            return new T5PromptStrategy(Map.of(PromptTask.EXPLAIN_CODE, "explain java"));
        }
        if (id.contains("flan-t5") || id.contains("t5-small") || id.contains("google-t5") || id.contains("t5")) {
            return new T5PromptStrategy(Map.of(
                    PromptTask.SUMMARIZE, "summarize",
                    PromptTask.TRANSLATE_TO_GERMAN, "translate English to German",
                    PromptTask.TRANSLATE_TO_ENGLISH, "translate German to English"));
        }
        return RawPromptStrategy.INSTANCE;
    }

    /**
     * Tasks supported by {@code modelId}, in declaration order of
     * {@link PromptTask}, for stable UI presentation.
     */
    public static List<PromptTask> supportedTasks(String modelId) {
        PromptStrategy strategy = forModel(modelId);
        return java.util.Arrays.stream(PromptTask.values())
                .filter(strategy.supportedTasks()::contains)
                .toList();
    }

    private static PromptStrategy chatMl(EnumSet<PromptTask> tasks) {
        return new ChatMlPromptStrategy(ChatTaskInstructions.standard(), tasks);
    }
}
