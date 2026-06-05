package com.aresstack.windirectml.inference.smollm2;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class SmolLM2WarpRuntimeTest {

    @TempDir
    Path tempDir;

    @Test
    void defaultWarpRuntimeReportsMissingNativeExecutor() throws Exception {
        try (SmolLM2WarpRuntime runtime = SmolLM2WarpRuntime.prepare(createExecutablePackage(), 16)) {
            assertFalse(runtime.executable());
            assertTrue(runtime.status().reason().contains(SmolLM2WarpExecutorFactory.EXECUTOR_CLASS_PROPERTY));
            assertFalse(runtime.plan().bufferPlan().entries().isEmpty());
            assertFalse(runtime.uploadManifest().weightEntries().isEmpty());
            assertEquals(runtime.plan().bufferPlan().totalWeightBytes(), runtime.uploadManifest().totalUploadBytes());

            SmolLM2RuntimeUnsupportedException exception = assertThrows(SmolLM2RuntimeUnsupportedException.class,
                    () -> runtime.generateTokenIds(new SmolLM2TokenRuntimeRequest(List.of(0), 1, null)));
            assertTrue(exception.getMessage().contains(SmolLM2WarpExecutorFactory.EXECUTOR_CLASS_PROPERTY));
        }
    }

    @Test
    void autoRuntimeFallsBackToReferenceWhenWarpExecutorIsUnavailable() throws Exception {
        try (SmolLM2Runtime runtime = SmolLM2Runtime.loadAuto(createExecutablePackage(), null, 16)) {
            assertEquals(SmolLM2RuntimeMode.REFERENCE, runtime.runtimeMode());
            assertTrue(runtime.warpExecutionStatus().isEmpty());
        }
    }

    @Test
    void explicitWarpRuntimeUsesInjectedExecutorWhenAvailable() throws Exception {
        ImmediateTokenWarpExecutor executor = new ImmediateTokenWarpExecutor();

        try (SmolLM2Runtime runtime = SmolLM2Runtime.loadWarp(createExecutablePackage(), null, 16, executor)) {
            SmolLM2TokenRuntimeResult result = runtime.generateTokenIds(
                    new SmolLM2TokenRuntimeRequest(List.of(4, 5), 1, null));

            assertEquals(SmolLM2RuntimeMode.WARP, runtime.runtimeMode());
            assertEquals(List.of(7), result.generatedTokenIds());
            assertEquals(List.of(4, 5, 7), result.fullTokenIds());
            assertEquals("length", result.finishReason());
            assertTrue(runtime.warpExecutionStatus().orElseThrow().executable());
            assertFalse(executor.closed());
        }
        assertTrue(executor.closed());
    }

    @Test
    void autoRuntimeSelectsWarpWhenInjectedExecutorIsAvailable() throws Exception {
        try (SmolLM2Runtime runtime = SmolLM2Runtime.loadAuto(
                createExecutablePackage(), null, 16, new ImmediateTokenWarpExecutor())) {
            assertEquals(SmolLM2RuntimeMode.WARP, runtime.runtimeMode());
        }
    }

    @Test
    void warpRuntimeRejectsManifestOnlyPackage() throws Exception {
        SmolLM2Config config = SmolLM2TestFixtures.config135(false);
        SmolLM2LayoutReport report = new SmolLM2LayoutValidator().validate(
                config, SmolLM2TestFixtures.completeCatalog(config, true));
        Path output = tempDir.resolve("manifest-only.wdmlpack");
        new SmolLM2WdmlPackManifestWriter().writeManifestOnly(output, config, report);

        SmolLM2RuntimeUnsupportedException exception = assertThrows(SmolLM2RuntimeUnsupportedException.class,
                () -> SmolLM2WarpRuntime.prepare(SmolLM2RuntimePackage.open(output), 16));

        assertEquals(SmolLM2LayoutReport.MISSING_DENSE_PAYLOAD, exception.getMessage());
    }

    private SmolLM2RuntimePackage createExecutablePackage() throws Exception {
        SmolLM2Config config = SmolLM2TestFixtures.config135(false);
        SmolLM2TestFixtures.writeModelDirectory(tempDir, config, true);
        Path output = tempDir.resolve("model-" + System.nanoTime() + ".wdmlpack");
        new SmolLM2WdmlPackCompiler().compile(new SmolLM2CompileOptions(tempDir, output, false, false));
        return SmolLM2RuntimePackage.open(output);
    }

    private static final class ImmediateTokenWarpExecutor implements SmolLM2WarpExecutor {

        private SmolLM2WarpSession session;

        @Override
        public SmolLM2WarpSession openSession(SmolLM2Weights weights, SmolLM2WarpRuntimePlan plan) {
            session = SmolLM2WarpExecutor.super.openSession(weights, plan);
            return session;
        }

        @Override
        public SmolLM2WarpExecutionStatus inspect(SmolLM2WarpRuntimePlan plan) {
            return new SmolLM2WarpExecutionStatus(true, "warp", "test executor is available", List.of());
        }

        @Override
        public SmolLM2TokenRuntimeResult generate(SmolLM2Weights weights,
                                                  SmolLM2WarpRuntimePlan plan,
                                                  SmolLM2TokenRuntimeRequest request) {
            return new SmolLM2TokenRuntimeResult(
                    request.inputTokenIds(),
                    List.of(7),
                    List.of(4, 5, 7),
                    1,
                    "length",
                    request.maxNewTokens());
        }

        boolean closed() {
            return session != null && session.closed();
        }
    }
}
