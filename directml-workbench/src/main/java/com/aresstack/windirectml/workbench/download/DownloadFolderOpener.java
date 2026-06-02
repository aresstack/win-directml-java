package com.aresstack.windirectml.workbench.download;

import java.awt.Desktop;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.Consumer;

/**
 * Opens a local model target folder in the system file manager.
 * <p>
 * Uses {@link Desktop#open(File)} when supported, with a Windows fallback
 * using {@code explorer.exe}.
 */
public final class DownloadFolderOpener {

    private DownloadFolderOpener() {
    }

    /**
     * Opens the specified folder in the system file manager.
     * Creates the folder if it does not exist.
     *
     * @param folder the folder to open
     * @param logger callback for status/error messages
     */
    public static void openFolder(Path folder, Consumer<String> logger) {
        try {
            Files.createDirectories(folder);
        } catch (IOException e) {
            logger.accept("ERROR: Could not create folder: " + folder + " (" + e.getMessage() + ")");
            return;
        }

        if (Desktop.isDesktopSupported()) {
            Desktop desktop = Desktop.getDesktop();
            if (desktop.isSupported(Desktop.Action.OPEN)) {
                try {
                    desktop.open(folder.toFile());
                    logger.accept("Opened folder: " + folder);
                    return;
                } catch (IOException e) {
                    logger.accept("Warning: Desktop.open failed: " + e.getMessage()
                            + ". Trying fallback...");
                }
            }
        }

        // Windows fallback
        try {
            new ProcessBuilder("explorer.exe", folder.toAbsolutePath().toString())
                    .start();
            logger.accept("Opened folder (explorer.exe): " + folder);
        } catch (IOException e) {
            logger.accept("ERROR: Could not open folder: " + folder + " (" + e.getMessage() + ")");
        }
    }
}
