package com.aresstack.windirectml.inference.artifact;

import com.aresstack.windirectml.inference.gemma.Gemma3NativeWarpRuntime;
import com.aresstack.windirectml.inference.gemma.Gemma3RuntimePackage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-WORKBENCH-CONVERT-1: the Gemma 3 lifecycle exposes a real compiler so the Workbench Convert
 * button is enabled once the raw HF files are present, and it targets exactly {@code model_gemma3.wdmlpack}.
 * Device-free (no compile run here; the real compile→run path is covered by Gemma3NativeWarpRuntimeTest).
 */
class Gemma3PackageLifecycleTest {

    @TempDir
    Path tempDir;

    private final Gemma3PackageLifecycle lifecycle = new Gemma3PackageLifecycle();

    @Test
    void hasCompilerAndTargetsModelGemma3Wdmlpack() {
        assertTrue(lifecycle.hasCompiler(), "Gemma must report a compiler so Convert is offered");
        assertEquals("model_gemma3.wdmlpack", lifecycle.defaultPackagePath(tempDir).getFileName().toString());
    }

    @Test
    void rawCompleteWithoutPackageEnablesConvert() throws IOException {
        Files.writeString(tempDir.resolve("config.json"), "{}");
        Files.writeString(tempDir.resolve("model.safetensors"), "weights");

        ModelArtifactStatus status = lifecycle.inspect(tempDir);
        assertEquals(RawAssetState.RAW_VALID, status.rawState());
        assertEquals(PackageState.PACKAGE_MISSING, status.packageState());
        assertTrue(status.hasCompiler());
        assertFalse(status.ready());

        ModelConversionPlan plan = lifecycle.planConversion(tempDir);
        assertEquals(ModelConversionAction.CONVERT, plan.action(), "missing package + valid raw -> Convert");
        assertEquals(lifecycle.defaultPackagePath(tempDir), plan.targetPackagePath());
    }

    @Test
    void emptyDirPlansInspectNotConvert() {
        ModelConversionPlan plan = lifecycle.planConversion(tempDir);
        assertEquals(ModelConversionAction.INSPECT, plan.action(), "no raw source -> nothing to convert yet");
    }

    @Test
    void corruptPackagePlansRepair() throws IOException {
        Files.writeString(tempDir.resolve("config.json"), "{}");
        Files.writeString(tempDir.resolve("model.safetensors"), "weights");
        Files.writeString(tempDir.resolve("model_gemma3.wdmlpack"), "not a real package");

        ModelArtifactStatus status = lifecycle.inspect(tempDir);
        assertEquals(PackageState.PACKAGE_CORRUPT, status.packageState());

        ModelConversionPlan plan = lifecycle.planConversion(tempDir);
        assertEquals(ModelConversionAction.REPAIR, plan.action());
    }

    @Test
    void existingPackageMatchesOnlyTheExactName() throws IOException {
        assertTrue(lifecycle.existingPackage(tempDir).isEmpty());
        Files.writeString(tempDir.resolve("gemma3.wdmlpack"), "x");
        assertTrue(lifecycle.existingPackage(tempDir).isEmpty(), "only model_gemma3.wdmlpack counts");
        Files.writeString(tempDir.resolve("model_gemma3.wdmlpack"), "x");
        assertTrue(lifecycle.existingPackage(tempDir).isPresent());
        assertEquals("model_gemma3.wdmlpack",
                lifecycle.existingPackage(tempDir).get().getFileName().toString());
    }

    /**
     * Real convert (the UI Convert path) → {@code model_gemma3.wdmlpack} that the native runtime finds.
     * CPU-only (the compiler needs no GPU); gated on the local model. The real files are hard-linked into
     * a temp dir so AppData is untouched and {@code @TempDir} cleans up.
     */
    @Test
    void realConvertProducesRuntimeLoadablePackageTheRuntimeFinds() throws IOException {
        Path modelDir = resolveModelDir();
        assumeTrue(modelDir != null, "Real Gemma 3 270M model not present");

        Path src = tempDir.resolve("gemma-src");
        Files.createDirectories(src);
        linkOrCopy(modelDir.resolve("config.json"), src.resolve("config.json"));
        linkOrCopy(modelDir.resolve("model.safetensors"), src.resolve("model.safetensors"));
        linkOrCopy(modelDir.resolve("tokenizer.json"), src.resolve("tokenizer.json"));

        // Check is green (raw complete, package missing -> Convert offered).
        ModelConversionPlan plan = lifecycle.planConversion(src);
        assertEquals(ModelConversionAction.CONVERT, plan.action());

        ModelConversionResult result = lifecycle.convert(src, false);
        assertTrue(result.ok(), "convert should succeed: " + result.message());

        Path pkg = lifecycle.defaultPackagePath(src);
        assertEquals("model_gemma3.wdmlpack", pkg.getFileName().toString());
        assertTrue(Files.isRegularFile(pkg), "convert must write model_gemma3.wdmlpack");

        // The runtime finds exactly this file, and it is a runtime-loadable Gemma package.
        assertNull(Gemma3NativeWarpRuntime.describeMissingPackage(
                Gemma3NativeWarpRuntime.defaultPackagePath(src)), "runtime must find the package");
        assertTrue(Gemma3RuntimePackage.open(pkg).runtimeLoadable(), "package must be runtime-loadable");

        // After convert the lifecycle reports READY.
        assertTrue(lifecycle.inspect(src).ready(), "converted package -> ready");
    }

    private static void linkOrCopy(Path from, Path to) throws IOException {
        try {
            Files.createLink(to, from); // instant, no data copy, when on the same volume
        } catch (IOException | UnsupportedOperationException e) {
            Files.copy(from, to);
        }
    }

    private static Path resolveModelDir() {
        String override = System.getProperty("gemma.testModelDir");
        if (override != null && !override.isBlank()) {
            return dirIfValid(Path.of(override));
        }
        String appData = System.getenv("APPDATA");
        if (appData != null && !appData.isBlank()) {
            Path p = dirIfValid(Path.of(appData, ".directml", "model", "gemma-3-270m-it"));
            if (p != null) {
                return p;
            }
        }
        String home = System.getProperty("user.home");
        return home == null ? null : dirIfValid(Path.of(home, ".directml", "model", "gemma-3-270m-it"));
    }

    private static Path dirIfValid(Path dir) {
        return dir != null && Files.isRegularFile(dir.resolve("config.json"))
                && Files.isRegularFile(dir.resolve("model.safetensors"))
                && Files.isRegularFile(dir.resolve("tokenizer.json")) ? dir : null;
    }
}
