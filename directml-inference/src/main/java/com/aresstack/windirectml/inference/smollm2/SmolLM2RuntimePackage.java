package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.model.RuntimeModelPackage;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;

/**
 * SmolLM2 package facade that validates wdmlpack metadata without loading payloads.
 */
public final class SmolLM2RuntimePackage {

    private final RuntimeModelPackage modelPackage;
    private final boolean executable;

    private SmolLM2RuntimePackage(RuntimeModelPackage modelPackage, boolean executable) {
        this.modelPackage = Objects.requireNonNull(modelPackage, "modelPackage");
        this.executable = executable;
    }

    public static SmolLM2RuntimePackage open(Path packagePath) throws IOException {
        return from(RuntimeModelPackage.open(packagePath));
    }

    public static SmolLM2RuntimePackage from(RuntimeModelPackage modelPackage) throws IOException {
        validate(modelPackage.manifest());
        return new SmolLM2RuntimePackage(modelPackage, modelPackage.runtimeLoadable());
    }

    public RuntimeModelPackage modelPackage() {
        return modelPackage;
    }

    public boolean executable() {
        return executable;
    }

    public String runtimeLoadableReason() {
        Object value = modelPackage.manifest().get("runtimeLoadableReason");
        return value == null ? SmolLM2LayoutReport.RUNTIME_NOT_IMPLEMENTED : String.valueOf(value);
    }

    private static void validate(Map<String, Object> manifest) throws IOException {
        if (!"smollm2".equals(String.valueOf(manifest.get("modelFamily")))) {
            throw new IOException("Invalid SmolLM2 package: modelFamily must be smollm2");
        }
        if (!"llama-causal-decoder".equals(String.valueOf(manifest.get("architecture")))) {
            throw new IOException("Invalid SmolLM2 package: architecture must be llama-causal-decoder");
        }
    }
}
