package com.aresstack.windirectml.workbench.download;

import javax.swing.JOptionPane;
import java.awt.Component;
import java.awt.Desktop;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Locale;
import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Opens download URLs in the system default browser.
 */
public final class DownloadUrlOpener {

    private static final Logger LOG = Logger.getLogger(DownloadUrlOpener.class.getName());

    private DownloadUrlOpener() {
    }

    /**
     * Opens the supplied URL in the default browser and shows a warning dialog on failure.
     */
    public static void openInBrowser(String url, Component parent) {
        try {
            openInBrowser(url, DesktopBrowserLauncher.DEFAULT);
        } catch (IllegalArgumentException | IOException | UnsupportedOperationException ex) {
            LOG.log(Level.WARNING, "Could not open download URL in browser", ex);
            JOptionPane.showMessageDialog(parent,
                    "Could not open download URL in the default browser: " + ex.getMessage(),
                    "Browser unavailable",
                    JOptionPane.WARNING_MESSAGE);
        }
    }

    static void openInBrowser(String url, BrowserLauncher browserLauncher) throws IOException {
        Objects.requireNonNull(browserLauncher, "browserLauncher");
        browserLauncher.browse(toBrowserUri(url));
    }

    static URI toBrowserUri(String url) {
        String trimmed = url == null ? "" : url.trim();
        if (trimmed.isBlank()) {
            throw new IllegalArgumentException("URL must not be blank");
        }
        try {
            URI uri = new URI(trimmed);
            String scheme = uri.getScheme() == null ? "" : uri.getScheme().toLowerCase(Locale.ROOT);
            if (!scheme.equals("http") && !scheme.equals("https")) {
                throw new IllegalArgumentException("Only http/https URLs can be opened: " + trimmed);
            }
            return uri;
        } catch (URISyntaxException ex) {
            throw new IllegalArgumentException("Invalid URL: " + trimmed, ex);
        }
    }

    @FunctionalInterface
    interface BrowserLauncher {
        void browse(URI uri) throws IOException;
    }

    private static final class DesktopBrowserLauncher {
        private static final BrowserLauncher DEFAULT = uri -> {
            if (!Desktop.isDesktopSupported()) {
                throw new UnsupportedOperationException("Desktop integration is not supported");
            }
            Desktop desktop = Desktop.getDesktop();
            if (!desktop.isSupported(Desktop.Action.BROWSE)) {
                throw new UnsupportedOperationException("Desktop browser action is not supported");
            }
            desktop.browse(uri);
        };
    }
}
