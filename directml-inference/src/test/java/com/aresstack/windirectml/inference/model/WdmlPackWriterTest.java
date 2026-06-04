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
    void writesAndReadsPayloadPackageHeader() throws Exception {
        Path pack = tempDir.resolve("model-payload.wdmlpack");
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("format", "wdmlpack");
        manifest.put("version", WdmlPackWriter.VERSION);
        manifest.put("mode", "payload");
        manifest.put("payloadIncluded", true);
        manifest.put("tensors", List.of(Map.of(
                "name", "tiny",
                "dataType", 2,
                "byteLength", 4L,
                "payloadOffset", 0L,
                "payloadLength", 4L)));

        WdmlPackWriter.writeWithPayload(pack, manifest, List.of(
                new WdmlPackWriter.PayloadEntry("tiny", 0L, 4L, channel -> channel.write(java.nio.ByteBuffer.wrap(new byte[]{1, 2, 3, 4})))
        ), 4L);

        WdmlPackWriter.Header header = WdmlPackWriter.readHeader(pack);
        assertFalse(header.manifestOnly());
        assertTrue(header.payloadIncluded());
        assertEquals(4L, header.payloadLength());
        assertEquals(0L, header.payloadOffset() % WdmlPackWriter.PAYLOAD_ALIGNMENT);

        Map<String, Object> read = WdmlPackWriter.readManifest(pack);
        assertEquals("payload", read.get("mode"));
        assertEquals(true, read.get("payloadIncluded"));
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
