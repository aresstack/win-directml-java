package com.aresstack.windirectml.workbench;

import com.aresstack.windirectml.workbench.panels.*;

import javax.swing.*;

/**
 * Entry point for the DirectML Workbench – a Java 21 Swing demo that exercises
 * the {@code directml-runtime} public API directly (no JSON-RPC sidecar).
 */
public final class DirectMlWorkbenchApp {

    private DirectMlWorkbenchApp() {}

    public static void main(String[] args) {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception ignored) {}

        SwingUtilities.invokeLater(() -> {
            var frame = new WorkbenchFrame();
            frame.setVisible(true);
        });
    }
}
