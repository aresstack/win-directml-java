package com.aresstack.windirectml.inference.model;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class WdmlPackWriterTest {

    @TempDir
    Path tempDir;

    @Test
    void writesAndReadsManifestOnlyPackage() throws Exception {
        Path pack = tempDir.resolve("model.wdmlpack");
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("format", "wdmlpack");
        manifest.put("version", WdmlPackWriter.VERSION);
        manifest.put("mode", "manifest-only");
        manifest.put("tensors", List.of(Map.of(
                "name", "model.embed_tokens.weight",
                "dataType", 10,
                "byteLength", 272269312L,
                "payloadOffset", -1L)));

        WdmlPackWriter.writeManifestOnly(pack, manifest);

        assertTrue(Files.size(pack) > WdmlPackWriter.HEADER_SIZE);
        Map<String, Object> read = WdmlPackWriter.readManifest(pack);
        assertEquals("wdmlpack", read.get("format"));
        assertEquals("manifest-only", read.get("mode"));
        assertTrue(read.containsKey("tensors"));
    }

    @Test
    void rejectsInvalidMagic() throws Exception {
        Path file = tempDir.resolve("broken.wdmlpack");
        Files.write(file, new byte[WdmlPackWriter.HEADER_SIZE]);
        IOExceptionAssert.assertThrowsIOException(() -> WdmlPackWriter.readManifest(file));
    }

    /**
     * Avoids depending on JUnit's exact checked-exception overload inference.
     */
    private static final class IOExceptionAssert {
        static void assertThrowsIOException(ThrowingRunnable runnable) {
            assertThrows(java.io.IOException.class, runnable::run);
        }
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run() throws Exception;
    }
}
