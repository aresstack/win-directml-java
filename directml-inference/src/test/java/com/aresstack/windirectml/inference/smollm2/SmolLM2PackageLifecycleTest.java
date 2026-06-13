package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.artifact.ModelArtifactStatus;
import com.aresstack.windirectml.inference.artifact.PackageState;
import com.aresstack.windirectml.inference.artifact.RawAssetState;
import com.aresstack.windirectml.inference.artifact.SmolLM2PackageLifecycle;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * W2: the SmolLM2 lifecycle (which the workbench runner now delegates to) reports honest package
 * states and never compiles during inspection/validation.
 */
class SmolLM2PackageLifecycleTest {

    @TempDir
    Path tempDir;

    private final SmolLM2PackageLifecycle lifecycle = new SmolLM2PackageLifecycle();

    @Test
    void missingPackageIsNotReadyAndValidateThrowsActionableWithoutCompiling() throws Exception {
        SmolLM2Config config = SmolLM2TestFixtures.config135(false);
        SmolLM2TestFixtures.writeModelDirectory(tempDir, config, true); // valid raw, no package yet

        ModelArtifactStatus status = lifecycle.inspect(tempDir);
        assertEquals(RawAssetState.RAW_VALID, status.rawState());
        assertEquals(PackageState.PACKAGE_MISSING, status.packageState());
        assertFalse(status.ready());

        lifecycle.planConversion(tempDir);
        IllegalStateException error = assertThrows(IllegalStateException.class,
                () -> lifecycle.validateOrThrowBeforeInference(tempDir));
        assertTrue(error.getMessage().contains("Convert"), error.getMessage());

        assertFalse(Files.isRegularFile(tempDir.resolve("model.wdmlpack")),
                "inspect/plan/validate must never compile a package");
    }

    @Test
    void validPackageIsReadyAndValidatePasses() throws Exception {
        SmolLM2Config config = SmolLM2TestFixtures.config135(false);
        SmolLM2TestFixtures.writeModelDirectory(tempDir, config, true);
        new SmolLM2WdmlPackCompiler().compile(
                new SmolLM2CompileOptions(tempDir, tempDir.resolve("model.wdmlpack"), false, true));

        ModelArtifactStatus status = lifecycle.inspect(tempDir);
        assertEquals(PackageState.PACKAGE_VALID, status.packageState());
        assertTrue(status.executable());
        assertTrue(status.ready());

        lifecycle.validateOrThrowBeforeInference(tempDir); // must not throw
    }

    @Test
    void stalePackageIsNotReadyAndValidateThrows() throws Exception {
        SmolLM2Config config = SmolLM2TestFixtures.config135(false);

        // Build the package in dirA (which mmaps its SafeTensors), then inspect a sibling dirB whose
        // SafeTensors source differs - so the stored fingerprint no longer matches. Using two dirs
        // avoids the Windows lock on rewriting a just-compiled, memory-mapped SafeTensors file.
        Path dirA = Files.createDirectories(tempDir.resolve("a"));
        SmolLM2TestFixtures.writeModelDirectory(dirA, config, true);
        new SmolLM2WdmlPackCompiler().compile(
                new SmolLM2CompileOptions(dirA, dirA.resolve("model.wdmlpack"), false, true));

        Path dirB = Files.createDirectories(tempDir.resolve("b"));
        SmolLM2TestFixtures.writeModelDirectory(dirB, config, true);
        Path shard = dirB.resolve("model.safetensors");
        byte[] grown = new byte[(int) Files.size(shard) + 16];
        System.arraycopy(Files.readAllBytes(shard), 0, grown, 0, (int) Files.size(shard));
        Files.write(shard, grown);
        Files.copy(dirA.resolve("model.wdmlpack"), dirB.resolve("model.wdmlpack"));

        ModelArtifactStatus status = lifecycle.inspect(dirB);
        assertEquals(PackageState.PACKAGE_STALE, status.packageState());
        assertFalse(status.ready());
        assertThrows(IllegalStateException.class, () -> lifecycle.validateOrThrowBeforeInference(dirB));
    }
}
