package com.aresstack.windirectml.inference.artifact;

import java.io.IOException;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Default {@link ModelArtifactService} backed by per-family {@link ModelPackageLifecycle} adapters.
 * Conversion failures are turned into a {@link ModelConversionResult} rather than propagated, so the
 * UI/library can report them uniformly; {@link #convert} remains the only write path.
 */
public final class DefaultModelArtifactService implements ModelArtifactService {

    private final Map<ModelFamily, ModelPackageLifecycle> lifecycles;

    public DefaultModelArtifactService(Map<ModelFamily, ModelPackageLifecycle> lifecycles) {
        this.lifecycles = new EnumMap<>(Objects.requireNonNull(lifecycles, "lifecycles"));
    }

    /**
     * Service wired with the standard lifecycles: real compilers for Qwen/SmolLM2/T5 and explicit
     * direct-source legacy markers for embedding/reranker/Phi-3.
     */
    public static DefaultModelArtifactService createDefault() {
        Map<ModelFamily, ModelPackageLifecycle> map = new EnumMap<>(ModelFamily.class);
        map.put(ModelFamily.QWEN, new QwenPackageLifecycle());
        map.put(ModelFamily.SMOLLM2, new SmolLM2PackageLifecycle());
        map.put(ModelFamily.T5, new T5PackageLifecycle());
        map.put(ModelFamily.GEMMA3, new Gemma3DownloadLifecycle());
        map.put(ModelFamily.EMBEDDING, new CompilerMissingLifecycle(ModelFamily.EMBEDDING,
                List.of("config.json", "tokenizer.json"),
                List.of(List.of("*.safetensors", "pytorch_model.bin"))));
        map.put(ModelFamily.RERANKER, new CompilerMissingLifecycle(ModelFamily.RERANKER,
                List.of("config.json", "tokenizer.json"),
                List.of(List.of("*.safetensors", "pytorch_model.bin"))));
        map.put(ModelFamily.PHI3, new CompilerMissingLifecycle(ModelFamily.PHI3,
                List.of("config.json"),
                List.of(List.of("*.onnx", "model.safetensors"))));
        return new DefaultModelArtifactService(map);
    }

    @Override
    public ModelPackageLifecycle lifecycle(ModelFamily family) {
        ModelPackageLifecycle lifecycle = lifecycles.get(Objects.requireNonNull(family, "family"));
        if (lifecycle == null) {
            throw new IllegalArgumentException("No artifact lifecycle registered for family " + family);
        }
        return lifecycle;
    }

    @Override
    public ModelArtifactStatus inspect(ModelFamily family, Path modelDir) {
        return lifecycle(family).inspect(modelDir);
    }

    @Override
    public ModelConversionPlan planConversion(ModelFamily family, Path modelDir) {
        return lifecycle(family).planConversion(modelDir);
    }

    @Override
    public ModelConversionResult convert(ModelFamily family, Path modelDir, boolean force) {
        ModelPackageLifecycle lifecycle = lifecycle(family);
        if (!lifecycle.hasCompiler()) {
            return ModelConversionResult.failed(
                    "Package compiler not implemented for " + family.displayName()
                            + ". This model is downloadable but not executable until a wdmlpack compiler exists.");
        }
        try {
            return lifecycle.convert(modelDir, force);
        } catch (IOException | RuntimeException e) {
            return ModelConversionResult.failed(
                    family.displayName() + " conversion failed: " + e.getMessage());
        }
    }

    @Override
    public void validateOrThrowBeforeInference(ModelFamily family, Path modelDir) {
        lifecycle(family).validateOrThrowBeforeInference(modelDir);
    }
}
