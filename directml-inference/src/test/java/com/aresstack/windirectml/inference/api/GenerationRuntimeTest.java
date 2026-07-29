package com.aresstack.windirectml.inference.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.aresstack.windirectml.catalog.CatalogBackend;
import com.aresstack.windirectml.catalog.CatalogModelFamily;
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
    void nonGenerationFamilyWithPresentPackageIsUnsupportedFamily(@TempDir Path dir) throws IOException {
        // Encoder families are not generation families and have no generation adapter. Lay down a
        // stub package so the package-only presence check passes; family dispatch is next.
        LocalRuntimeModelDescriptor encoder = descriptor("sentence-transformers/all-MiniLM-L6-v2");
        Files.createFile(dir.resolve(encoder.runtimePackageFileName()));
        GenerationException ex = assertThrows(GenerationException.class,
                () -> GenerationRuntime.load(encoder, dir, CatalogBackend.CPU, LoadPolicy.PACKAGE_ONLY));
        // Proves ordering: backend + dir + package all pass, then family lookup fails cleanly.
        assertEquals(GenerationErrorCode.UNSUPPORTED_FAMILY, ex.errorCode());
    }

    @Test
    void allowCompilePolicySkipsPackagePresenceCheck(@TempDir Path dir) {
        LocalRuntimeModelDescriptor encoder = descriptor("sentence-transformers/all-MiniLM-L6-v2");
        // No package on disk, but ALLOW_COMPILE must not raise PACKAGE_MISSING; it proceeds to
        // family lookup (encoder has no generation adapter) -> UNSUPPORTED_FAMILY.
        GenerationException ex = assertThrows(GenerationException.class,
                () -> GenerationRuntime.load(encoder, dir, CatalogBackend.CPU, LoadPolicy.ALLOW_COMPILE));
        assertEquals(GenerationErrorCode.UNSUPPORTED_FAMILY, ex.errorCode());
    }

    @Test
    void registeredGenerationFamilyDispatchesToAdapterAndFailsAtInit(@TempDir Path dir)
            throws IOException {
        // Qwen IS a registered generation family. With a present-but-empty package and no model
        // assets, dispatch reaches the adapter and the engine fails to initialize — proving the
        // adapter is wired (INITIALIZATION_FAILED), not an UNSUPPORTED_FAMILY short-circuit.
        LocalRuntimeModelDescriptor qwen = descriptor("Qwen/Qwen2.5-Coder-0.5B-Instruct");
        Files.createFile(dir.resolve(qwen.runtimePackageFileName()));
        GenerationException ex = assertThrows(GenerationException.class,
                () -> GenerationRuntime.load(qwen, dir, CatalogBackend.CPU, LoadPolicy.PACKAGE_ONLY));
        assertEquals(GenerationErrorCode.INITIALIZATION_FAILED, ex.errorCode());
    }

    @Test
    void gemmaWarpDispatchesToAdapterWithoutPython(@TempDir Path dir) throws IOException {
        // Gemma is gated (real run needs an HF token); this contract test needs no model. With a
        // present package + tokenizer, a WARP load must dispatch to the native Gemma adapter and
        // return a handle (family GEMMA3, backend WARP) — proving it is wired and never routes to a
        // Python/CPU path. The native device is only touched on generate(), not on open().
        LocalRuntimeModelDescriptor gemma = descriptor("google/gemma-3-270m-it");
        Files.createFile(dir.resolve(gemma.runtimePackageFileName()));
        Files.createFile(dir.resolve("tokenizer.json"));
        try (GenerationModelHandle handle =
                GenerationRuntime.load(gemma, dir, CatalogBackend.WARP, LoadPolicy.PACKAGE_ONLY)) {
            assertEquals(CatalogModelFamily.GEMMA3, handle.family());
            assertEquals(CatalogBackend.WARP, handle.backend());
        }
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
