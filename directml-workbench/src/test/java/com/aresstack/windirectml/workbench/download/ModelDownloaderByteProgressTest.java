package com.aresstack.windirectml.workbench.download;

import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelDownloaderByteProgressTest {

    @TempDir
    Path tempDir;

    @Test
    void reportsByteProgressWithinLargeManifestFile() throws Exception {
        byte[] payload = new byte[3 * 1024 * 1024 + 17];
        for (int i = 0; i < payload.length; i++) {
            payload[i] = (byte) (i & 0x7f);
        }

        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/model.safetensors", exchange -> {
            exchange.getResponseHeaders().add("Content-Type", "application/octet-stream");
            exchange.sendResponseHeaders(200, payload.length);
            try (var out = exchange.getResponseBody()) {
                out.write(payload);
            }
        });
        server.start();
        try {
            String url = "http://127.0.0.1:" + server.getAddress().getPort() + "/model.safetensors";
            ModelDownloadManifest manifest = new ModelDownloadManifest(
                    "test-model",
                    "test-model",
                    List.of(new ModelFileDescriptor(
                            "Model",
                            true,
                            url,
                            url,
                            "model.safetensors"
                    )));
            List<ModelDownloader.ProgressEvent> events = new ArrayList<>();

            ModelDownloader.downloadFromManifest(manifest, tempDir, false, ignored -> {
            }, events::add);

            assertEquals(payload.length, Files.size(tempDir.resolve("model.safetensors")));
            assertFalse(events.isEmpty(), "Expected progress events");
            assertTrue(events.stream().anyMatch(e -> !e.completed()
                            && !e.skipped()
                            && e.bytesRead() > 0
                            && e.totalBytes() == payload.length
                            && e.aggregatePercent() > 0
                            && e.aggregatePercent() < 100),
                    "Expected in-file byte progress event, got: " + events);
            assertTrue(events.stream().anyMatch(e -> e.completed()
                            && !e.skipped()
                            && e.bytesRead() == payload.length
                            && e.totalBytes() == payload.length
                            && e.aggregatePercent() == 100),
                    "Expected completed progress event, got: " + events);
        } finally {
            server.stop(0);
        }
    }

    @Test
    void reportsSkippedExistingFileAsCompletedProgress() throws IOException, InterruptedException {
        Path target = tempDir.resolve("model.safetensors");
        Files.write(target, new byte[]{1, 2, 3});
        ModelDownloadManifest manifest = new ModelDownloadManifest(
                "test-model",
                "test-model",
                List.of(new ModelFileDescriptor(
                        "Model",
                        true,
                        "https://example.com/model.safetensors",
                        "https://example.com/model.safetensors",
                        "model.safetensors"
                )));
        List<ModelDownloader.ProgressEvent> events = new ArrayList<>();

        ModelDownloader.downloadFromManifest(manifest, tempDir, false, ignored -> {
        }, events::add);

        assertEquals(1, events.size());
        ModelDownloader.ProgressEvent event = events.get(0);
        assertTrue(event.completed());
        assertTrue(event.skipped());
        assertEquals(100, event.aggregatePercent());
    }
}
