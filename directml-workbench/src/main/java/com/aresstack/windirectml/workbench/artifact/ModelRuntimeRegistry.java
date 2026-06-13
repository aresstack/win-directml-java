package com.aresstack.windirectml.workbench.artifact;

import com.aresstack.windirectml.inference.artifact.ModelFamily;
import com.aresstack.windirectml.inference.artifact.QwenPackageLifecycle;
import com.aresstack.windirectml.workbench.WorkbenchModel;
import com.aresstack.windirectml.workbench.download.QwenModelDownloadConfig;

import java.nio.file.Path;

/**
 * Builds {@link ModelRuntimeDescriptor}s from the live {@link WorkbenchModel}, so DownloadPanel and the
 * runtime tabs share one path/lifecycle source per model. Descriptors are built on demand because some
 * paths depend on mutable UI state (the selected Qwen ONNX variant).
 */
public final class ModelRuntimeRegistry {

    /** Generation model id used across the Workbench for Qwen 0.5B. */
    public static final String QWEN_05B_MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct";

    private final WorkbenchModel model;

    public ModelRuntimeRegistry(WorkbenchModel model) {
        this.model = model;
    }

    /**
     * Qwen 0.5B runtime descriptor. The supported path is the q4f16/ONNX source in the directml-int4
     * directory, converted to {@code model_q4f16.wdmlpack}, which the runtime then loads exclusively.
     * The DownloadPanel row and {@code QwenInferenceEngine} both resolve this same directory + package.
     */
    public ModelRuntimeDescriptor qwen05b() {
        return new ModelRuntimeDescriptor(
                ModelFamily.QWEN,
                QWEN_05B_MODEL_ID,
                "Qwen2.5-Coder 0.5B Instruct (q4f16)",
                this::qwen05bRuntimeDir,
                () -> new QwenPackageLifecycle(model.getQwenModelFile()),
                ModelRuntimeDescriptor.RuntimeEntryPoint.SUMMARIZER);
    }

    /** The single Qwen 0.5B runtime directory (directml-int4) used for both convert and load. */
    public Path qwen05bRuntimeDir() {
        return model.getModelRoot().resolve(QwenModelDownloadConfig.LOCAL_DIR_NAME);
    }
}
