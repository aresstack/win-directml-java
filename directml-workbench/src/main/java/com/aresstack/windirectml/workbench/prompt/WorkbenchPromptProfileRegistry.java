package com.aresstack.windirectml.workbench.prompt;

import com.aresstack.windirectml.config.generation.GenerationModelRegistry;

import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Resolves model ids to model-family specific Workbench prompt profiles.
 */
public final class WorkbenchPromptProfileRegistry {

    private WorkbenchPromptProfileRegistry() {
    }

    public static List<WorkbenchPromptTemplate> templatesFor(String modelId) {
        return profileFor(modelId).supportedTemplates();
    }

    public static RenderedWorkbenchPrompt render(String modelId,
                                                 WorkbenchPromptTemplate template,
                                                 String inputText) {
        return profileFor(modelId).render(template, inputText);
    }

    public static WorkbenchPromptProfile profileFor(String modelId) {
        String normalized = normalize(modelId);
        if (normalized.contains("qwen2.5-coder")) {
            return PromptProfiles.qwenCoder();
        }
        if (normalized.contains("smollm2")) {
            return PromptProfiles.smolLm2Instruct();
        }
        if (normalized.contains("flan-t5")) {
            return PromptProfiles.flanT5();
        }
        if (normalized.contains("codet5-base-multi-sum")) {
            return PromptProfiles.codeT5MultiSum();
        }
        if (normalized.contains("codet5-small")) {
            return PromptProfiles.codeT5Small();
        }
        if (normalized.contains("t5-small") || normalized.contains("google-t5")) {
            return PromptProfiles.googleT5();
        }

        GenerationModelRegistry.Entry entry = GenerationModelRegistry.findByModelId(modelId);
        if (entry != null && entry.architecture() == GenerationModelRegistry.Architecture.CAUSAL_LM) {
            return PromptProfiles.rawOnly();
        }
        return PromptProfiles.rawOnly();
    }

    private static String normalize(String modelId) {
        return Objects.toString(modelId, "").toLowerCase(Locale.ROOT);
    }
}
