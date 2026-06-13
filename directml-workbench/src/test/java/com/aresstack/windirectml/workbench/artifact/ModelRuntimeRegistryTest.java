package com.aresstack.windirectml.workbench.artifact;

import com.aresstack.windirectml.inference.artifact.ModelFamily;
import com.aresstack.windirectml.inference.qwen.QwenWdmlPackCompileTool;
import com.aresstack.windirectml.workbench.WorkbenchModel;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Qwen 0.5B regression: the DownloadPanel descriptor and the runtime must resolve the SAME package
 * path (directml-int4 / model_q4f16.wdmlpack). This is the consistency the artifact lifecycle requires.
 */
class ModelRuntimeRegistryTest {

    @TempDir
    Path tempDir;

    @Test
    void qwen05bDescriptorResolvesTheRuntimePackagePathTheEngineLoads() {
        WorkbenchModel model = new WorkbenchModel();
        model.setModelRoot(tempDir);

        ModelRuntimeDescriptor qwen = new ModelRuntimeRegistry(model).qwen05b();
        assertEquals(ModelFamily.QWEN, qwen.family());

        Path runtimeDir = tempDir.resolve("qwen2.5-coder-0.5b-directml-int4");
        assertEquals(runtimeDir, qwen.rawDir(), "descriptor must point at the directml-int4 dir");

        // The package the runtime loader resolves for the same dir + selected ONNX variant.
        Path engineResolved = QwenWdmlPackCompileTool.resolvePackagePath(runtimeDir, model.getQwenModelFile());
        assertEquals(engineResolved, qwen.runtimePackagePath(),
                "panel descriptor package path must equal the runtime package path");
        assertTrue(qwen.runtimePackagePath().getFileName().toString().endsWith(".wdmlpack"));
    }

    @Test
    void qwen05bIsNotReadyUntilThePackageIsBuilt() {
        WorkbenchModel model = new WorkbenchModel();
        model.setModelRoot(tempDir);
        ModelRuntimeDescriptor qwen = new ModelRuntimeRegistry(model).qwen05b();

        // No package on disk -> not ready, and inspect must not create one.
        assertFalse(qwen.inspect().ready());
    }
}
