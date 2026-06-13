package com.aresstack.windirectml.inference.artifact;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit coverage for the unified artifact lifecycle state machine, the legacy direct-source path,
 * and the strict contract that {@code inspect}/{@code plan}/{@code validate} never write while
 * {@code convert} is the only write path.
 */
class DefaultModelArtifactServiceTest {

    @TempDir
    Path tempDir;

    // --- planConversion action mapping -------------------------------------------------------

    @Test
    void missingPackageWithValidRawPlansConvert() {
        ModelConversionPlan plan = planFor(status(RawAssetState.RAW_VALID, PackageState.PACKAGE_MISSING, false));
        assertEquals(ModelConversionAction.CONVERT, plan.action());
        assertTrue(plan.action().writesPackage());
    }

    @Test
    void stalePackagePlansReconvert() {
        ModelConversionPlan plan = planFor(status(RawAssetState.RAW_VALID, PackageState.PACKAGE_STALE, false));
        assertEquals(ModelConversionAction.RECONVERT, plan.action());
    }

    @Test
    void corruptOrNotExecutablePackagePlansRepair() {
        assertEquals(ModelConversionAction.REPAIR,
                planFor(status(RawAssetState.RAW_VALID, PackageState.PACKAGE_CORRUPT, false)).action());
        assertEquals(ModelConversionAction.REPAIR,
                planFor(status(RawAssetState.RAW_VALID, PackageState.PACKAGE_NOT_EXECUTABLE, false)).action());
        assertEquals(ModelConversionAction.REPAIR,
                planFor(status(RawAssetState.RAW_VALID, PackageState.PACKAGE_NOT_LOADABLE, false)).action());
    }

    @Test
    void validPackagePlansReconvert() {
        ModelConversionPlan plan = planFor(status(RawAssetState.RAW_VALID, PackageState.PACKAGE_VALID, true));
        assertEquals(ModelConversionAction.RECONVERT, plan.action());
    }

    @Test
    void missingRawAndNoUsablePackagePlansInspectOnly() {
        ModelConversionPlan plan = planFor(status(RawAssetState.RAW_MISSING, PackageState.PACKAGE_MISSING, false));
        assertEquals(ModelConversionAction.INSPECT, plan.action());
        assertFalse(plan.action().writesPackage());
    }

    @Test
    void familyWithoutCompilerPlansNotSupported() {
        FakeLifecycle fake = new FakeLifecycle(false,
                status(RawAssetState.RAW_VALID, PackageState.PACKAGE_COMPILER_MISSING, false));
        assertEquals(ModelConversionAction.NOT_SUPPORTED, fake.planConversion(tempDir).action());
    }

    // --- ready() derivation ------------------------------------------------------------------

    @Test
    void readyRequiresValidExecutablePackageForCompilerFamilies() {
        assertTrue(status(RawAssetState.RAW_VALID, PackageState.PACKAGE_VALID, true).ready());
        assertFalse(status(RawAssetState.RAW_VALID, PackageState.PACKAGE_VALID, false).ready());
        assertFalse(status(RawAssetState.RAW_VALID, PackageState.PACKAGE_MISSING, false).ready());
    }

    @Test
    void compilerMissingFamilyIsNeverReady() {
        assertFalse(status(RawAssetState.RAW_VALID, PackageState.PACKAGE_COMPILER_MISSING, false).ready());
        assertFalse(status(RawAssetState.RAW_INCOMPLETE, PackageState.PACKAGE_COMPILER_MISSING, false).ready());
    }

    // --- compiler-missing lifecycle + real raw inspection ------------------------------------

    @Test
    void compilerMissingLifecycleReportsRawStateButIsNotExecutableAndNeverWrites() throws IOException {
        CompilerMissingLifecycle lifecycle = new CompilerMissingLifecycle(ModelFamily.EMBEDDING,
                List.of("config.json", "tokenizer.json"),
                List.of(List.of("*.safetensors", "pytorch_model.bin")));

        Path missingDir = tempDir.resolve("nope");
        assertEquals(RawAssetState.RAW_MISSING, lifecycle.inspect(missingDir).rawState());

        Path dir = Files.createDirectories(tempDir.resolve("e5"));
        assertEquals(RawAssetState.RAW_INCOMPLETE, lifecycle.inspect(dir).rawState());

        Files.writeString(dir.resolve("config.json"), "{}");
        Files.writeString(dir.resolve("tokenizer.json"), "{}");
        Files.writeString(dir.resolve("model.safetensors"), "x");
        ModelArtifactStatus valid = lifecycle.inspect(dir);
        assertEquals(RawAssetState.RAW_VALID, valid.rawState());
        assertEquals(PackageState.PACKAGE_COMPILER_MISSING, valid.packageState());
        assertFalse(valid.ready(), "no compiler -> not executable even with valid raw files");

        assertFalse(hasPackageFile(dir), "inspect must not write a package");
        assertFalse(lifecycle.convert(dir, true).ok(), "convert is unsupported when no compiler exists");
        assertFalse(hasPackageFile(dir), "compiler-missing convert must not write a package");
    }

    @Test
    void compilerFamilyInspectOnEmptyDirIsMissingPackageAndWritesNothing() {
        ModelArtifactService service = DefaultModelArtifactService.createDefault();
        for (ModelFamily family : List.of(ModelFamily.QWEN, ModelFamily.SMOLLM2, ModelFamily.T5)) {
            ModelArtifactStatus status = service.inspect(family, tempDir);
            assertEquals(PackageState.PACKAGE_MISSING, status.packageState(), family + " package state");
            assertFalse(status.ready(), family + " must not be ready");
            assertFalse(hasPackageFile(tempDir), family + " inspect must not write");
        }
    }

    // --- service-level contract --------------------------------------------------------------

    @Test
    void serviceConvertOnLegacyFamilyFailsWithoutWriting() {
        ModelArtifactService service = DefaultModelArtifactService.createDefault();
        ModelConversionResult result = service.convert(ModelFamily.EMBEDDING, tempDir, true);
        assertFalse(result.ok());
        assertTrue(result.message().contains("not implemented"));
        assertFalse(hasPackageFile(tempDir));
    }

    @Test
    void onlyConvertWritesAcrossLifecycleOperations() throws IOException {
        AtomicInteger writes = new AtomicInteger();
        FakeLifecycle fake = new FakeLifecycle(true,
                status(RawAssetState.RAW_VALID, PackageState.PACKAGE_MISSING, false)) {
            @Override
            public ModelConversionResult convert(Path modelDir, boolean force) {
                writes.incrementAndGet();
                return new ModelConversionResult(true, modelDir.resolve("model.wdmlpack"), "wrote");
            }
        };
        fake.inspect(tempDir);
        fake.planConversion(tempDir);
        assertThrows(IllegalStateException.class, () -> fake.validateOrThrowBeforeInference(tempDir));
        assertEquals(0, writes.get(), "inspect/plan/validate must not write");
        fake.convert(tempDir, false);
        assertEquals(1, writes.get(), "convert is the only write path");
    }

    @Test
    void validateBeforeInferencePassesWhenReadyAndThrowsOtherwise() {
        FakeLifecycle ready = new FakeLifecycle(true,
                status(RawAssetState.RAW_VALID, PackageState.PACKAGE_VALID, true));
        ready.validateOrThrowBeforeInference(tempDir); // no throw

        FakeLifecycle notReady = new FakeLifecycle(true,
                status(RawAssetState.RAW_VALID, PackageState.PACKAGE_MISSING, false));
        IllegalStateException error = assertThrows(IllegalStateException.class,
                () -> notReady.validateOrThrowBeforeInference(tempDir));
        assertTrue(error.getMessage().contains("Convert"), "actionable message points at Convert flow");
    }

    // --- helpers -----------------------------------------------------------------------------

    private ModelArtifactStatus status(RawAssetState raw, PackageState pkg, boolean executable) {
        boolean hasCompiler = pkg != PackageState.PACKAGE_COMPILER_MISSING;
        return new ModelArtifactStatus(ModelFamily.SMOLLM2, tempDir, raw, pkg, executable, hasCompiler, "test");
    }

    private ModelConversionPlan planFor(ModelArtifactStatus status) {
        return new FakeLifecycle(true, status).planConversion(tempDir);
    }

    private boolean hasPackageFile(Path dir) {
        if (!Files.isDirectory(dir)) {
            return false;
        }
        try (var stream = Files.list(dir)) {
            return stream.anyMatch(p -> p.getFileName().toString().endsWith(".wdmlpack"));
        } catch (IOException e) {
            return false;
        }
    }

    /** Minimal in-test lifecycle returning a fixed status, used to exercise the shared default logic. */
    private static class FakeLifecycle implements ModelPackageLifecycle {
        private final boolean hasCompiler;
        private final ModelArtifactStatus status;

        FakeLifecycle(boolean hasCompiler, ModelArtifactStatus status) {
            this.hasCompiler = hasCompiler;
            this.status = status;
        }

        @Override
        public ModelFamily family() {
            return status.family();
        }

        @Override
        public boolean hasCompiler() {
            return hasCompiler;
        }

        @Override
        public Path defaultPackagePath(Path modelDir) {
            return modelDir.resolve("model.wdmlpack");
        }

        @Override
        public ModelArtifactStatus inspect(Path modelDir) {
            return status;
        }

        @Override
        public ModelConversionResult convert(Path modelDir, boolean force) {
            return new ModelConversionResult(true, defaultPackagePath(modelDir), "fake");
        }
    }
}
