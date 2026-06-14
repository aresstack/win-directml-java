package com.aresstack.windirectml.workbench.artifact;

import com.aresstack.windirectml.inference.artifact.CompilerMissingLifecycle;
import com.aresstack.windirectml.inference.artifact.Gemma3DownloadLifecycle;
import com.aresstack.windirectml.inference.artifact.ModelArtifactStatus;
import com.aresstack.windirectml.inference.artifact.ModelConversionResult;
import com.aresstack.windirectml.inference.artifact.ModelFamily;
import com.aresstack.windirectml.inference.artifact.ModelPackageLifecycle;
import com.aresstack.windirectml.inference.artifact.PackageState;
import com.aresstack.windirectml.inference.artifact.RawAssetState;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** W4: the Swing-free row controller derives the right Convert label and honours the write contract. */
class ModelArtifactRowTest {

    @TempDir
    Path tempDir;

    @Test
    void missingPackagePlansConvertAndRefreshWritesNothing() {
        AtomicInteger writes = new AtomicInteger();
        ModelArtifactRow row = row(writes, true,
                status(RawAssetState.RAW_VALID, PackageState.PACKAGE_MISSING, false));

        ModelArtifactRow.RowView view = row.refresh();
        assertEquals("Convert", view.convertLabel());
        assertTrue(view.convertEnabled());
        assertFalse(view.ready());
        assertEquals(0, writes.get(), "refresh/inspect/plan must not write");
    }

    @Test
    void validPackagePlansReconvertAndIsReady() {
        ModelArtifactRow row = row(new AtomicInteger(), true,
                status(RawAssetState.RAW_VALID, PackageState.PACKAGE_VALID, true));
        ModelArtifactRow.RowView view = row.refresh();
        assertEquals("Reconvert", view.convertLabel());
        assertTrue(view.convertEnabled());
        assertTrue(view.ready());
    }

    @Test
    void notExecutablePlansRepair() {
        ModelArtifactRow row = row(new AtomicInteger(), true,
                status(RawAssetState.RAW_VALID, PackageState.PACKAGE_NOT_EXECUTABLE, false));
        assertEquals("Repair package", row.refresh().convertLabel());
    }

    @Test
    void rawMissingShowsDownloadFirstDisabled() {
        ModelArtifactRow row = row(new AtomicInteger(), true,
                status(RawAssetState.RAW_MISSING, PackageState.PACKAGE_MISSING, false));
        ModelArtifactRow.RowView view = row.refresh();
        assertEquals("Download first", view.convertLabel());
        assertFalse(view.convertEnabled());
    }

    @Test
    void compilerMissingShowsNoFakeConvertAndConvertFails() {
        AtomicInteger writes = new AtomicInteger();
        ModelArtifactRow row = row(writes, false,
                status(RawAssetState.RAW_VALID, PackageState.PACKAGE_COMPILER_MISSING, false));
        ModelArtifactRow.RowView view = row.refresh();
        assertEquals("Compiler missing", view.convertLabel());
        assertFalse(view.convertEnabled());
        assertFalse(view.ready());
        assertTrue(view.statusText().contains("package compiler not implemented"),
                "status must stay honest about the missing compiler: " + view.statusText());
        assertTrue(view.convertTooltip().contains("Package compiler not implemented"),
                "tooltip must explain the not-executable state: " + view.convertTooltip());
        assertFalse(row.convert().ok());
        assertEquals(0, writes.get(), "compiler-missing convert must not write");
    }


    @Test
    void gemmaDownloadOnlyRowDoesNotReportCompilerImplementationError() throws IOException {
        Files.writeString(tempDir.resolve("model.safetensors"), "weights");
        Files.writeString(tempDir.resolve("config.json"), "{}");
        Files.writeString(tempDir.resolve("tokenizer.json"), "{}");

        ModelArtifactRow row = new ModelArtifactRow(ModelFamily.GEMMA3, () -> tempDir, Gemma3DownloadLifecycle::new);

        ModelArtifactRow.RowView view = row.refresh();
        assertEquals("Download only", view.convertLabel());
        assertFalse(view.convertEnabled());
        assertFalse(view.ready());
        assertTrue(view.statusText().contains("Gemma 3 files are present"));
        assertFalse(view.statusText().contains("package compiler not implemented"));
        assertTrue(view.convertTooltip().contains("download/probe only"));
    }

    @Test
    void convertIsTheOnlyWritePath() {
        AtomicInteger writes = new AtomicInteger();
        ModelArtifactRow row = row(writes, true,
                status(RawAssetState.RAW_VALID, PackageState.PACKAGE_MISSING, false));
        row.inspect();
        row.plan();
        row.refresh();
        assertEquals(0, writes.get());
        assertTrue(row.convert().ok());
        assertEquals(1, writes.get(), "convert is the only write path");
    }

    @Test
    void realCompilerMissingLifecycleRowWritesNothingAndIsNotExecutable() throws IOException {
        ModelArtifactRow row = new ModelArtifactRow(ModelFamily.EMBEDDING, () -> tempDir,
                () -> new CompilerMissingLifecycle(ModelFamily.EMBEDDING,
                        List.of("config.json", "tokenizer.json"),
                        List.of(List.of("*.safetensors", "pytorch_model.bin"))));

        ModelArtifactRow.RowView view = row.refresh();
        assertEquals("Compiler missing", view.convertLabel());
        assertFalse(view.ready());
        assertFalse(row.convert().ok());
        assertFalse(hasPackage(tempDir), "compiler-missing row must never write a package");
    }

    private ModelArtifactRow row(AtomicInteger writes, boolean hasCompiler, ModelArtifactStatus status) {
        return new ModelArtifactRow(status.family(), () -> tempDir,
                () -> new FakeLifecycle(hasCompiler, status, writes));
    }

    private ModelArtifactStatus status(RawAssetState raw, PackageState pkg, boolean executable) {
        return new ModelArtifactStatus(ModelFamily.SMOLLM2, tempDir, raw, pkg, executable,
                pkg != PackageState.PACKAGE_COMPILER_MISSING, "reason");
    }

    private static boolean hasPackage(Path dir) throws IOException {
        try (var stream = Files.list(dir)) {
            return stream.anyMatch(p -> p.getFileName().toString().endsWith(".wdmlpack"));
        }
    }

    private static final class FakeLifecycle implements ModelPackageLifecycle {
        private final boolean hasCompiler;
        private final ModelArtifactStatus status;
        private final AtomicInteger writes;

        FakeLifecycle(boolean hasCompiler, ModelArtifactStatus status, AtomicInteger writes) {
            this.hasCompiler = hasCompiler;
            this.status = status;
            this.writes = writes;
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
            writes.incrementAndGet();
            return new ModelConversionResult(true, defaultPackagePath(modelDir), "fake convert");
        }
    }
}
