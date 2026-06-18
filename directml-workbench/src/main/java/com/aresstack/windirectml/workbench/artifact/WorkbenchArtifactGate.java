package com.aresstack.windirectml.workbench.artifact;

import com.aresstack.windirectml.inference.artifact.DefaultModelArtifactService;
import com.aresstack.windirectml.inference.artifact.ModelArtifactService;
import com.aresstack.windirectml.inference.artifact.ModelArtifactStatus;
import com.aresstack.windirectml.inference.artifact.ModelFamily;
import com.aresstack.windirectml.inference.artifact.ModelPackageLifecycle;
import com.aresstack.windirectml.inference.artifact.Phi3PackageLifecycle;
import com.aresstack.windirectml.inference.artifact.QwenPackageLifecycle;
import com.aresstack.windirectml.inference.artifact.SmolLM2PackageLifecycle;
import com.aresstack.windirectml.inference.artifact.T5PackageLifecycle;
import com.aresstack.windirectml.encoder.pack.EncoderPackageLifecycle;

import java.nio.file.Path;
import java.util.EnumMap;
import java.util.Map;

/**
 * Single entry point every Workbench runtime tab calls before inference. It routes through the
 * central {@link ModelArtifactService} so no tab loads SafeTensors/ONNX directly: families without a
 * wdmlpack compiler (encoder/reranker/Phi-3 today) fail fast here with an actionable message instead
 * of silently running from raw weights.
 */
public final class WorkbenchArtifactGate {

    private static final ModelArtifactService SERVICE = buildService();

    private WorkbenchArtifactGate() {
    }

    /**
     * Service wired with the real encoder/reranker lifecycles (so embeddings/reranker are executable
     * from a wdmlpack) plus the generation families. Phi-3 is now compiler-backed (Convert ->
     * model_phi3.wdmlpack); its Workbench runtime status stays PLANNED until a heap-light runtime loader lands.
     */
    private static ModelArtifactService buildService() {
        Map<ModelFamily, ModelPackageLifecycle> map = new EnumMap<>(ModelFamily.class);
        map.put(ModelFamily.QWEN, new QwenPackageLifecycle());
        map.put(ModelFamily.SMOLLM2, new SmolLM2PackageLifecycle());
        map.put(ModelFamily.T5, new T5PackageLifecycle());
        map.put(ModelFamily.EMBEDDING, EncoderPackageLifecycle.embedding());
        map.put(ModelFamily.RERANKER, EncoderPackageLifecycle.reranker());
        map.put(ModelFamily.PHI3, new Phi3PackageLifecycle());
        return new DefaultModelArtifactService(map);
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
