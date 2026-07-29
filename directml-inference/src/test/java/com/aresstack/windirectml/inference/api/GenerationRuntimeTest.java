package com.aresstack.windirectml.inference.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.aresstack.windirectml.catalog.CatalogBackend;
import com.aresstack.windirectml.catalog.LocalModelCatalog;
import com.aresstack.windirectml.catalog.LocalRuntimeModelDescriptor;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Fail-closed validation contract of {@link GenerationRuntime}. These tests need no real model: they
 * drive the catalog-descriptor validation, backend-matrix enforcement, and package-only presence
 * checks that run before any family adapter is invoked.
 */
class GenerationRuntimeTest {

    private static LocalRuntimeModelDescriptor descriptor(String repo) {
        LocalRuntimeModelDescriptor d = LocalModelCatalog.findByRepositoryId(repo);
        assertNotNull(d, "catalog is missing " + repo);
        return d;
    }

    @Test
    void gemmaWithCpuIsUnsupportedBackend(@TempDir Path dir) {
        LocalRuntimeModelDescriptor gemma = descriptor("google/gemma-3-270m-it");
        GenerationException ex = assertThrows(GenerationException.class,
                () -> GenerationRuntime.load(gemma, dir, CatalogBackend.CPU));
        assertEquals(GenerationErrorCode.UNSUPPORTED_BACKEND, ex.errorCode());
        // Gemma is native WARP/DirectML only — the matrix must not admit CPU (no Python fallback).
        assertTrue(gemma.supportedBackends().contains(CatalogBackend.WARP));
        assertTrue(gemma.supportedBackends().contains(CatalogBackend.AUTO));
        assertTrue(!gemma.supportedBackends().contains(CatalogBackend.CPU));
    }

    @Test
    void phi3WithWarpIsUnsupportedBackend(@TempDir Path dir) {
        LocalRuntimeModelDescriptor phi3 = descriptor("microsoft/Phi-3-mini-4k-instruct-onnx");
        GenerationException ex = assertThrows(GenerationException.class,
                () -> GenerationRuntime.load(phi3, dir, CatalogBackend.WARP));
        assertEquals(GenerationErrorCode.UNSUPPORTED_BACKEND, ex.errorCode());
        // Phi-3 has no WARP mode in the code — the matrix must not fake one.
        assertTrue(!phi3.supportedBackends().contains(CatalogBackend.WARP));
        assertTrue(phi3.supportedBackends().contains(CatalogBackend.CPU));
        assertTrue(phi3.supportedBackends().contains(CatalogBackend.DIRECTML));
    }

    @Test
    void missingModelDirectoryIsTyped() {
        LocalRuntimeModelDescriptor qwen = descriptor("Qwen/Qwen2.5-Coder-0.5B-Instruct");
        Path missing = Path.of("Z:/does/not/exist/win-directml-test-" + qwen.runtimeDirectoryName());
        GenerationException ex = assertThrows(GenerationException.class,
                () -> GenerationRuntime.load(qwen, missing, CatalogBackend.CPU));
        assertEquals(GenerationErrorCode.MODEL_DIRECTORY_NOT_FOUND, ex.errorCode());
    }

    @Test
    void packageOnlyWithoutPackageIsMissing(@TempDir Path dir) {
        LocalRuntimeModelDescriptor qwen = descriptor("Qwen/Qwen2.5-Coder-0.5B-Instruct");
        // Directory exists but holds no model_*.wdmlpack -> package-only must fail closed.
        GenerationException ex = assertThrows(GenerationException.class,
                () -> GenerationRuntime.load(qwen, dir, CatalogBackend.CPU, LoadPolicy.PACKAGE_ONLY));
        assertEquals(GenerationErrorCode.PACKAGE_MISSING, ex.errorCode());
    }

    @Test
    void presentPackageButNoAdapterYetIsUnsupportedFamily(@TempDir Path dir) throws IOException {
        LocalRuntimeModelDescriptor qwen = descriptor("Qwen/Qwen2.5-Coder-0.5B-Instruct");
        // Lay down a stub package so the package-only presence check passes; family dispatch is next.
        Files.createFile(dir.resolve(qwen.runtimePackageFileName()));
        GenerationException ex = assertThrows(GenerationException.class,
                () -> GenerationRuntime.load(qwen, dir, CatalogBackend.CPU, LoadPolicy.PACKAGE_ONLY));
        // W2 registers no adapters yet (W3 wires them); this proves ordering: backend + dir + package
        // all pass, then family lookup fails cleanly rather than NPE-ing.
        assertEquals(GenerationErrorCode.UNSUPPORTED_FAMILY, ex.errorCode());
    }

    @Test
    void allowCompilePolicySkipsPackagePresenceCheck(@TempDir Path dir) {
        LocalRuntimeModelDescriptor qwen = descriptor("Qwen/Qwen2.5-Coder-0.5B-Instruct");
        // No package on disk, but ALLOW_COMPILE must not raise PACKAGE_MISSING; it proceeds to
        // family lookup (which is empty in W2) -> UNSUPPORTED_FAMILY.
        GenerationException ex = assertThrows(GenerationException.class,
                () -> GenerationRuntime.load(qwen, dir, CatalogBackend.CPU, LoadPolicy.ALLOW_COMPILE));
        assertEquals(GenerationErrorCode.UNSUPPORTED_FAMILY, ex.errorCode());
    }

    @Test
    void requestBuilderRejectsBlankPromptAndBadTokenCount() {
        GenerationException blank = assertThrows(GenerationException.class,
                () -> GenerationRequest.of("   "));
        assertEquals(GenerationErrorCode.INVALID_REQUEST, blank.errorCode());

        GenerationException zeroTokens = assertThrows(GenerationException.class,
                () -> GenerationRequest.builder("hello").maxNewTokens(0).build());
        assertEquals(GenerationErrorCode.INVALID_REQUEST, zeroTokens.errorCode());
    }
}
