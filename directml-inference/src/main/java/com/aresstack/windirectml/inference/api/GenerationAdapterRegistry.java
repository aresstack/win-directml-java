package com.aresstack.windirectml.inference.api;

import com.aresstack.windirectml.catalog.CatalogModelFamily;
import java.util.ArrayList;
import java.util.Collection;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Resolves a {@link CatalogModelFamily} to its {@link FamilyGenerationAdapter}. The default set of
 * adapters is the built-in family list in {@link #defaults()}; a custom set can be supplied for
 * tests.
 */
final class GenerationAdapterRegistry {

    private final Map<CatalogModelFamily, FamilyGenerationAdapter> byFamily;

    GenerationAdapterRegistry(Collection<FamilyGenerationAdapter> adapters) {
        EnumMap<CatalogModelFamily, FamilyGenerationAdapter> map =
                new EnumMap<>(CatalogModelFamily.class);
        for (FamilyGenerationAdapter adapter : adapters) {
            FamilyGenerationAdapter previous = map.put(adapter.family(), adapter);
            if (previous != null) {
                throw new IllegalStateException(
                        "Duplicate generation adapter for family " + adapter.family());
            }
        }
        this.byFamily = map;
    }

    Optional<FamilyGenerationAdapter> find(CatalogModelFamily family) {
        return Optional.ofNullable(byFamily.get(family));
    }

    /**
     * The built-in adapters shipped with the runtime. Family adapters are registered here (W3);
     * causal/decoder-only families (Qwen, SmolLM2), Gemma&nbsp;3, Phi-3, and the seq2seq T5/CodeT5
     * family each contribute one entry.
     */
    static GenerationAdapterRegistry defaults() {
        List<FamilyGenerationAdapter> adapters = new ArrayList<>();
        // InferenceEngine-based families (uniform CatalogBackend.name().toLowerCase() mapping).
        adapters.add(new QwenGenerationAdapter());
        adapters.add(new T5GenerationAdapter());
        adapters.add(new Phi3GenerationAdapter());
        // SMOLLM2 and GEMMA3 have bespoke runtimes (not InferenceEngine) whose orchestration still
        // lives in the workbench runners (SmolLM2WorkbenchRuntimeRunner, Gemma3 native-warp routing).
        // Their adapters are added once that logic is relocated into this neutral module (W3/W4);
        // until then GenerationRuntime reports UNSUPPORTED_FAMILY for them rather than guessing.
        return new GenerationAdapterRegistry(adapters);
    }
}
