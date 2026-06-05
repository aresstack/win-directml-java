package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.model.RuntimeModelPackage;
import com.aresstack.windirectml.inference.model.RuntimeTensorCatalog;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * SmolLM2 package facade that validates wdmlpack metadata and resolves payload weights when present.
 */
public final class SmolLM2RuntimePackage {

    private final RuntimeModelPackage modelPackage;
    private final SmolLM2Config config;
    private final Optional<SmolLM2Weights> weights;
    private final boolean executable;

    private SmolLM2RuntimePackage(RuntimeModelPackage modelPackage,
                                  SmolLM2Config config,
                                  Optional<SmolLM2Weights> weights,
                                  boolean executable) {
        this.modelPackage = Objects.requireNonNull(modelPackage, "modelPackage");
        this.config = Objects.requireNonNull(config, "config");
        this.weights = Objects.requireNonNull(weights, "weights");
        this.executable = executable;
    }

    public static SmolLM2RuntimePackage open(Path packagePath) throws IOException {
        return from(RuntimeModelPackage.open(packagePath));
    }

    public static SmolLM2RuntimePackage from(RuntimeModelPackage modelPackage) throws IOException {
        validate(modelPackage.manifest());
        SmolLM2Config config = SmolLM2PackageMetadata.fromManifest(modelPackage.manifest()).toConfig();
        Optional<SmolLM2Weights> weights = resolveWeightsIfPresent(modelPackage, config);
        return new SmolLM2RuntimePackage(modelPackage, config, weights, modelPackage.runtimeLoadable());
    }

    public RuntimeModelPackage modelPackage() {
        return modelPackage;
    }

    public SmolLM2Config config() {
        return config;
    }

    public boolean weightsAvailable() {
        return weights.isPresent();
    }

    public Optional<SmolLM2Weights> weights() {
        return weights;
    }

    public SmolLM2Weights requireWeights() {
        return weights.orElseThrow(() -> new IllegalStateException(
                "SmolLM2 package does not contain loadable weights: " + modelPackage.packagePath()));
    }

    public boolean executable() {
        return executable;
    }

    public String runtimeLoadableReason() {
        Object value = modelPackage.manifest().get("runtimeLoadableReason");
        return value == null ? SmolLM2LayoutReport.RUNTIME_NOT_IMPLEMENTED : String.valueOf(value);
    }

    private static Optional<SmolLM2Weights> resolveWeightsIfPresent(RuntimeModelPackage modelPackage,
                                                                    SmolLM2Config config) throws IOException {
        if (!modelPackage.payloadIncluded()) {
            return Optional.empty();
        }
        RuntimeTensorCatalog catalog = modelPackage.runtimeTensorCatalog();
        return Optional.of(new SmolLM2WeightResolver().resolve(config, catalog));
    }

    private static void validate(Map<String, Object> manifest) throws IOException {
        if (!"smollm2".equals(String.valueOf(manifest.get("modelFamily")))) {
            throw new IOException("Invalid SmolLM2 package: modelFamily must be smollm2");
        }
        if (!"llama-causal-decoder".equals(String.valueOf(manifest.get("architecture")))) {
            throw new IOException("Invalid SmolLM2 package: architecture must be llama-causal-decoder");
        }
        if (!manifest.containsKey("model")) {
            throw new IOException("Invalid SmolLM2 package: missing model metadata");
        }
    }
}
