package com.aresstack.windirectml.workbench.download;

import org.junit.jupiter.api.Test;

import java.net.URI;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class DownloadUrlOpenerTest {

    @Test
    void acceptsHttpsUrlsForBrowserOpening() {
        URI uri = DownloadUrlOpener.toBrowserUri(" https://huggingface.co/model/file.safetensors ");

        assertEquals("https", uri.getScheme());
        assertEquals("huggingface.co", uri.getHost());
        assertEquals("/model/file.safetensors", uri.getPath());
    }

    @Test
    void rejectsNonHttpSchemes() {
        assertThrows(IllegalArgumentException.class,
                () -> DownloadUrlOpener.toBrowserUri("file:///C:/tmp/model.safetensors"));
    }

    @Test
    void delegatesOpeningToBrowserLauncher() throws Exception {
        AtomicReference<URI> opened = new AtomicReference<>();

        DownloadUrlOpener.openInBrowser("https://example.test/model.bin", opened::set);

        assertEquals(URI.create("https://example.test/model.bin"), opened.get());
    }
}
