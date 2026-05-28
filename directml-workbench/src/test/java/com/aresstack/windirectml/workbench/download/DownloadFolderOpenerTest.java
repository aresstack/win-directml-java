package com.aresstack.windirectml.workbench.download;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link DownloadFolderOpener} – verifies folder creation logic.
 * <p>
 * Note: actual Desktop.open / explorer.exe calls are not tested here because
 * CI is headless. We verify the folder-creation prerequisite and error handling.
 */
class DownloadFolderOpenerTest {

    @TempDir
    Path tempDir;

    @Test
    void createsDirectoryIfNotExists() {
        Path nested = tempDir.resolve("a/b/c");
        assertFalse(Files.exists(nested));

        List<String> log = new ArrayList<>();
        DownloadFolderOpener.openFolder(nested, log::add);

        assertTrue(Files.isDirectory(nested), "Folder should be created");
        // Should have at least one log message (opened or error)
        assertFalse(log.isEmpty());
    }

    @Test
    void existingDirectoryDoesNotFail() {
        // tempDir already exists
        List<String> log = new ArrayList<>();
        DownloadFolderOpener.openFolder(tempDir, log::add);

        assertTrue(Files.isDirectory(tempDir));
        assertFalse(log.isEmpty());
    }
}
