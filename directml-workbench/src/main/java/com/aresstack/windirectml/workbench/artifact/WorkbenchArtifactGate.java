package com.aresstack.windirectml.workbench.artifact;

import com.aresstack.windirectml.inference.artifact.DefaultModelArtifactService;
import com.aresstack.windirectml.inference.artifact.ModelArtifactService;
import com.aresstack.windirectml.inference.artifact.ModelArtifactStatus;
import com.aresstack.windirectml.inference.artifact.ModelFamily;

import java.nio.file.Path;

/**
 * Single entry point every Workbench runtime tab calls before inference. It routes through the
 * central {@link ModelArtifactService} so no tab loads SafeTensors/ONNX directly: families without a
 * wdmlpack compiler (encoder/reranker/Phi-3 today) fail fast here with an actionable message instead
 * of silently running from raw weights.
 */
public final class WorkbenchArtifactGate {

    private static final ModelArtifactService SERVICE = DefaultModelArtifactService.createDefault();

    private WorkbenchArtifactGate() {
    }

    /** Throws {@link IllegalStateException} with actionable guidance when the model is not executable. */
    public static void requireExecutablePackage(ModelFamily family, Path modelDir) {
        SERVICE.validateOrThrowBeforeInference(family, modelDir);
    }

    /** Read-only status (never writes/compiles). */
    public static ModelArtifactStatus inspect(ModelFamily family, Path modelDir) {
        return SERVICE.inspect(family, modelDir);
    }
}
