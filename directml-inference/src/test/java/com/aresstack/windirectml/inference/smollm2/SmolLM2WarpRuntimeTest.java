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
    void probeWarpRuntimeReportsWarpProbeWithoutExecutableKernels() throws Exception {
        try (SmolLM2WarpRuntime runtime = SmolLM2WarpRuntime.prepare(
                createExecutablePackage(), 16, new SmolLM2DirectMlWarpExecutor())) {
            assertFalse(runtime.executable());
            assertFalse(runtime.status().reason().isBlank());
            assertFalse(runtime.plan().bufferPlan().entries().isEmpty());
            assertFalse(runtime.uploadManifest().weightEntries().isEmpty());
            assertEquals(runtime.plan().bufferPlan().totalWeightBytes(), runtime.uploadManifest().totalUploadBytes());

            SmolLM2RuntimeUnsupportedException exception = assertThrows(SmolLM2RuntimeUnsupportedException.class,
                    () -> runtime.generateTokenIds(new SmolLM2TokenRuntimeRequest(List.of(0), 1, null)));
            assertEquals(runtime.status().reason(), exception.getMessage());
        }
    }

    @Test
    void autoRuntimeFallsBackToReferenceWhenWarpExecutorIsUnavailable() throws Exception {
        try (SmolLM2Runtime runtime = SmolLM2Runtime.loadAuto(createExecutablePackage(), null, 16,
                new SmolLM2UnsupportedWarpExecutor("warp executor unavailable for test"))) {
            assertEquals(SmolLM2RuntimeMode.REFERENCE, runtime.runtimeMode());
            assertTrue(runtime.warpExecutionStatus().isEmpty());
        }
    }

    @Test
    void autoRuntimeFallsBackToReferenceWhenWarpEngineFailsAtFirstUse() throws Exception {
        LazyFailingWarpExecutor executor = new LazyFailingWarpExecutor("warp engine init failed for test");
        try (SmolLM2Runtime runtime = SmolLM2Runtime.loadAuto(createExecutablePackage(), null, 16, executor)) {
            // Readiness probe is optimistic, so AUTO initially selects WARP.
            assertEquals(SmolLM2RuntimeMode.WARP, runtime.runtimeMode());

            SmolLM2TokenRuntimeResult result = runtime.generateTokenIds(
                    new SmolLM2TokenRuntimeRequest(List.of(4, 5), 1, null));

            // The lazy WARP failure must transparently fall back to the reference path and report it honestly.
            assertEquals(SmolLM2RuntimeMode.REFERENCE, runtime.runtimeMode());
            assertEquals(1, result.tokensGenerated());
            assertTrue(runtime.warpFallbackReason().isPresent());
            assertTrue(runtime.warpFallbackReason().orElseThrow().contains("warp engine init failed for test"));
        }
    }

    @Test
    void explicitWarpRuntimePropagatesEngineFailureWithoutFallback() throws Exception {
        LazyFailingWarpExecutor executor = new LazyFailingWarpExecutor("warp engine init failed for test");
        try (SmolLM2Runtime runtime = SmolLM2Runtime.loadWarp(createExecutablePackage(), null, 16, executor)) {
            assertThrows(SmolLM2RuntimeUnsupportedException.class,
                    () -> runtime.generateTokenIds(new SmolLM2TokenRuntimeRequest(List.of(4, 5), 1, null)));
            // Explicit WARP must not silently fall back to the reference path.
            assertEquals(SmolLM2RuntimeMode.WARP, runtime.runtimeMode());
        }
    }

    @Test
    void warpRuntimeForwardsStreamingConsumerToSession() throws Exception {
        StreamingWarpExecutor executor = new StreamingWarpExecutor();
        try (SmolLM2WarpRuntime runtime = SmolLM2WarpRuntime.prepare(createExecutablePackage(), 16, executor)) {
            List<Integer> streamed = new java.util.ArrayList<>();
            runtime.generateTokenIds(new SmolLM2TokenRuntimeRequest(List.of(4, 5), 3, null), streamed::add);
            assertEquals(List.of(11, 12, 13), streamed,
                    "The streaming consumer must be forwarded through the WARP runtime to the session");
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

    /**
     * Executor whose readiness probe is optimistic (executable) but whose session fails when the engine is first used —
     * mirrors the native session's lazy device/upload initialisation failing on first generate.
     */
    private static final class LazyFailingWarpExecutor implements SmolLM2WarpExecutor {
        private final String reason;

        private LazyFailingWarpExecutor(String reason) {
            this.reason = reason;
        }

        @Override
        public SmolLM2WarpSession openSession(SmolLM2Weights weights, SmolLM2WarpRuntimePlan plan) {
            return new LazyFailingWarpSession(plan, reason);
        }

        @Override
        public SmolLM2WarpExecutionStatus inspect(SmolLM2WarpRuntimePlan plan) {
            return new SmolLM2WarpExecutionStatus(true, "warp", "optimistic readiness for test", List.of());
        }

        @Override
        public SmolLM2TokenRuntimeResult generate(SmolLM2Weights weights,
                                                  SmolLM2WarpRuntimePlan plan,
                                                  SmolLM2TokenRuntimeRequest request) {
            throw new SmolLM2RuntimeUnsupportedException(reason);
        }
    }

    private static final class LazyFailingWarpSession implements SmolLM2WarpSession {
        private final SmolLM2WarpRuntimePlan plan;
        private final String reason;
        private boolean closed;

        private LazyFailingWarpSession(SmolLM2WarpRuntimePlan plan, String reason) {
            this.plan = plan;
            this.reason = reason;
        }

        @Override
        public SmolLM2WarpRuntimePlan plan() {
            return plan;
        }

        @Override
        public SmolLM2WarpUploadManifest uploadManifest() {
            return SmolLM2WarpUploadManifest.fromPlan(plan);
        }

        @Override
        public SmolLM2WarpExecutionStatus status() {
            return new SmolLM2WarpExecutionStatus(true, "warp", "optimistic readiness for test", List.of());
        }

        @Override
        public SmolLM2TokenRuntimeResult generateTokenIds(SmolLM2TokenRuntimeRequest request,
                                                          java.util.function.IntConsumer generatedTokenConsumer) {
            throw new SmolLM2RuntimeUnsupportedException(reason);
        }

        @Override
        public boolean closed() {
            return closed;
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    /**
     * Executor whose session streams a fixed set of tokens through the consumer, used to verify the consumer is
     * forwarded from {@link SmolLM2WarpRuntime} down to the session.
     */
    private static final class StreamingWarpExecutor implements SmolLM2WarpExecutor {
        @Override
        public SmolLM2WarpSession openSession(SmolLM2Weights weights, SmolLM2WarpRuntimePlan plan) {
            return new StreamingWarpSession(plan);
        }

        @Override
        public SmolLM2WarpExecutionStatus inspect(SmolLM2WarpRuntimePlan plan) {
            return new SmolLM2WarpExecutionStatus(true, "warp", "streaming test executor", List.of());
        }

        @Override
        public SmolLM2TokenRuntimeResult generate(SmolLM2Weights weights,
                                                  SmolLM2WarpRuntimePlan plan,
                                                  SmolLM2TokenRuntimeRequest request) {
            throw new UnsupportedOperationException("not used in this test");
        }
    }

    private static final class StreamingWarpSession implements SmolLM2WarpSession {
        private static final List<Integer> TOKENS = List.of(11, 12, 13);
        private final SmolLM2WarpRuntimePlan plan;
        private boolean closed;

        private StreamingWarpSession(SmolLM2WarpRuntimePlan plan) {
            this.plan = plan;
        }

        @Override
        public SmolLM2WarpRuntimePlan plan() {
            return plan;
        }

        @Override
        public SmolLM2WarpUploadManifest uploadManifest() {
            return SmolLM2WarpUploadManifest.fromPlan(plan);
        }

        @Override
        public SmolLM2WarpExecutionStatus status() {
            return new SmolLM2WarpExecutionStatus(true, "warp", "streaming test session", List.of());
        }

        @Override
        public SmolLM2TokenRuntimeResult generateTokenIds(SmolLM2TokenRuntimeRequest request,
                                                          java.util.function.IntConsumer generatedTokenConsumer) {
            List<Integer> full = new java.util.ArrayList<>(request.inputTokenIds());
            for (int token : TOKENS) {
                if (generatedTokenConsumer != null) {
                    generatedTokenConsumer.accept(token);
                }
                full.add(token);
            }
            return new SmolLM2TokenRuntimeResult(
                    request.inputTokenIds(),
                    TOKENS,
                    full,
                    TOKENS.size(),
                    "length",
                    request.maxNewTokens());
        }

        @Override
        public boolean closed() {
            return closed;
        }

        @Override
        public void close() {
            closed = true;
        }
    }
}
