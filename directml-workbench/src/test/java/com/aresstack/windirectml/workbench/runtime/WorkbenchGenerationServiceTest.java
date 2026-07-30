package com.aresstack.windirectml.workbench.runtime;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.aresstack.windirectml.catalog.CatalogBackend;
import com.aresstack.windirectml.inference.api.GenerationErrorCode;
import com.aresstack.windirectml.inference.api.GenerationException;
import com.aresstack.windirectml.inference.api.GenerationResult;
import com.aresstack.windirectml.inference.artifact.T5PackageLifecycle;
import com.aresstack.windirectml.runtime.facade.Backend;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Headless verification that the workbench drives generation through the shared neutral
 * {@link WorkbenchGenerationService} (i.e. the same {@code GenerationRuntime} AskAI uses), with no
 * per-family orchestration of its own. The fail-closed contracts need no model; the opt-in e2e drives
 * the full chain through the service with a real T5 model.
 */
class WorkbenchGenerationServiceTest {

    private final WorkbenchGenerationService service = new WorkbenchGenerationService();

    @Test
    void mapsFacadeBackendOntoCatalogBackend() {
        assertEquals(CatalogBackend.CPU, WorkbenchGenerationService.toCatalogBackend(Backend.CPU));
        assertEquals(CatalogBackend.WARP, WorkbenchGenerationService.toCatalogBackend(Backend.WARP));
        assertEquals(CatalogBackend.DIRECTML, WorkbenchGenerationService.toCatalogBackend(Backend.DIRECTML));
        assertEquals(CatalogBackend.AUTO, WorkbenchGenerationService.toCatalogBackend(Backend.AUTO));
        // HYBRID has no neutral equivalent -> AUTO ("best available").
        assertEquals(CatalogBackend.AUTO, WorkbenchGenerationService.toCatalogBackend(Backend.HYBRID));
    }

    @Test
    void unknownModelIsInvalidRequest(@TempDir Path dir) {
        GenerationException ex = assertThrows(GenerationException.class, () ->
                service.generate("not/a-real-model", dir, CatalogBackend.CPU, null, "hi", 8, null));
        assertEquals(GenerationErrorCode.INVALID_REQUEST, ex.errorCode());
    }

    @Test
    void gemmaWithCpuIsUnsupportedBackendNotPython(@TempDir Path dir) {
        // Gemma's only CPU path was an external Python bridge; through the shared runtime CPU is simply
        // outside the matrix, so the workbench never routes to Python.
        GenerationException ex = assertThrows(GenerationException.class, () ->
                service.generate("google/gemma-3-270m-it", dir, CatalogBackend.CPU, null, "hi", 8, null));
        assertEquals(GenerationErrorCode.UNSUPPORTED_BACKEND, ex.errorCode());
    }

    @Test
    void packageOnlyMissingIsTyped(@TempDir Path dir) {
        GenerationException ex = assertThrows(GenerationException.class, () ->
                service.generate("google-t5/t5-small", dir, CatalogBackend.CPU, null, "summarize: x", 8, null));
        assertEquals(GenerationErrorCode.PACKAGE_MISSING, ex.errorCode());
    }

    @Test
    void endToEndThroughServiceWithRealT5() throws IOException {
        String rawDirProp = System.getProperty("windirectml.wb.t5Dir");
        assumeTrue(rawDirProp != null && !rawDirProp.isBlank(),
                "windirectml.wb.t5Dir not set — opt-in workbench e2e");
        Path rawDir = Path.of(rawDirProp);
        assertTrue(Files.isDirectory(rawDir), "wb.t5Dir does not exist: " + rawDir);

        new T5PackageLifecycle().convert(rawDir, true);
        Path pkgDir = Files.createTempDirectory("wb-gen-t5-");
        try (Stream<Path> entries = Files.list(rawDir)) {
            List<Path> files = new ArrayList<>();
            entries.filter(Files::isRegularFile).forEach(files::add);
            for (Path f : files) {
                String n = f.getFileName().toString().toLowerCase(Locale.ROOT);
                if (!n.endsWith(".safetensors") && !n.endsWith(".bin")) {
                    Files.copy(f, pkgDir.resolve(f.getFileName().toString()));
                }
            }
        }
        AtomicInteger streamed = new AtomicInteger();
        GenerationResult result = service.generate("google-t5/t5-small", pkgDir, CatalogBackend.CPU, null,
                "summarize: Java is a programming language used for many applications.", 24,
                token -> streamed.incrementAndGet());
        assertNotNull(result);
        assertEquals(CatalogBackend.CPU, result.backend());
        assertTrue(result.text() != null && !result.text().trim().isEmpty(), "empty text");
        assertTrue(result.generatedTokenCount() > 0, "no tokens");
        assertTrue(streamed.get() > 0, "no streamed tokens");
    }
}
