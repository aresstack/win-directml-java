package com.aresstack.windirectml.workbench.runtime;

import com.aresstack.windirectml.inference.artifact.ModelArtifactStatus;
import com.aresstack.windirectml.inference.artifact.ModelConversionResult;
import com.aresstack.windirectml.inference.artifact.ModelFamily;
import com.aresstack.windirectml.inference.artifact.ModelPackageLifecycle;
import com.aresstack.windirectml.inference.artifact.PackageState;
import com.aresstack.windirectml.inference.artifact.RawAssetState;
import com.aresstack.windirectml.inference.prompt.PromptInput;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * W2: the SmolLM2 workbench runner never compiles a package. When the package is not ready, generate
 * fails fast with an actionable error and never invokes the lifecycle's conversion (write) path.
 */
class SmolLM2WorkbenchRuntimeRunnerTest {

    @TempDir
    Path tempDir;

    @Test
    void generateWithMissingPackageThrowsActionableAndNeverCompiles() {
        SpyLifecycle spy = new SpyLifecycle(tempDir,
                new ModelArtifactStatus(ModelFamily.SMOLLM2, tempDir, RawAssetState.RAW_VALID,
                        PackageState.PACKAGE_MISSING, false, true, "no package"));
        SmolLM2WorkbenchRuntimeRunner runner = new SmolLM2WorkbenchRuntimeRunner(tempDir, spy);

        IllegalStateException error = assertThrows(IllegalStateException.class,
                () -> runner.generate(PromptInput.raw("hello"), 4));

        assertTrue(error.getMessage().contains("Convert"), error.getMessage());
        assertEquals(0, spy.convertCalls, "generate must never invoke the compiler");
        assertFalse(Files.exists(tempDir.resolve("model.wdmlpack")), "generate must not write a package");
    }

    /** Lifecycle stub that records conversion attempts and returns a fixed (not-ready) status. */
    private static final class SpyLifecycle implements ModelPackageLifecycle {
        private final Path dir;
        private final ModelArtifactStatus status;
        int convertCalls;

        SpyLifecycle(Path dir, ModelArtifactStatus status) {
            this.dir = dir;
            this.status = status;
        }

        @Override
        public ModelFamily family() {
            return ModelFamily.SMOLLM2;
        }

        @Override
        public boolean hasCompiler() {
            return true;
        }

        @Override
        public Path defaultPackagePath(Path modelDir) {
            return dir.resolve("model.wdmlpack");
        }

        @Override
        public ModelArtifactStatus inspect(Path modelDir) {
            return status;
        }

        @Override
        public ModelConversionResult convert(Path modelDir, boolean force) {
            convertCalls++;
            return new ModelConversionResult(true, defaultPackagePath(modelDir), "spy");
        }
    }
}
