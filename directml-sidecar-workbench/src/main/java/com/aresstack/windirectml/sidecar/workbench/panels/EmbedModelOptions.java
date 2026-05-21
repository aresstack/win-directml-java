package com.aresstack.windirectml.sidecar.workbench.panels;

import com.aresstack.windirectml.config.models.EmbeddingModelRegistry;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Builds the option list for the Workbench {@code embed.model}
 * dropdown from the shared {@link EmbeddingModelRegistry}.
 *
 * <p>Only entries with {@code useCase == EMBEDDING} are exposed:
 * decoder, summarizer, reranker and unsupported entries from the
 * registry must never appear in the embedding selector because
 * choosing them would be rejected by the sidecar's {@code embed}
 * gate anyway (and the workbench should reflect that contract
 * up-front instead of letting the user assemble an invalid
 * command line).
 *
 * <p>The {@code minilm} / {@code e5} short aliases the sidecar
 * accepts on {@code -Dembed.model} are kept as the first two
 * options for backwards compatibility with users who scripted
 * the workbench, but they are now declared explicitly rather
 * than hard-coded as a string array.
 */
final class EmbedModelOptions {

    /** Family aliases historically supported by the Workbench dropdown. */
    static final String ALIAS_MINILM = "minilm";
    static final String ALIAS_E5 = "e5";

    private EmbedModelOptions() {
        // utility
    }

    /**
     * The full ordered list shown in the dropdown:
     * <ol>
     *   <li>{@code minilm}, {@code e5} family aliases (back-compat,
     *       always selectable),</li>
     *   <li>every {@code useCase=embedding} full model ID from the
     *       shared registry in declaration order.</li>
     * </ol>
     * Non-embedding registry entries (decoder, summarizer, &hellip;)
     * are deliberately omitted.
     */
    static List<String> embeddingOptions() {
        List<String> options = new ArrayList<String>();
        options.add(ALIAS_MINILM);
        options.add(ALIAS_E5);
        for (EmbeddingModelRegistry.Entry e : EmbeddingModelRegistry
                .entriesByUseCase(EmbeddingModelRegistry.UseCase.EMBEDDING)) {
            options.add(e.modelId());
        }
        return Collections.unmodifiableList(options);
    }
}
