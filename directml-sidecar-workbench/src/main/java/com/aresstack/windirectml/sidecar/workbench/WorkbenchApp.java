package com.aresstack.windirectml.sidecar.workbench;

import javax.swing.SwingUtilities;
import javax.swing.UIManager;
import javax.swing.UnsupportedLookAndFeelException;

/**
 * Entry point of the developer workbench. Pure Java 8 / Swing.
 *
 * <p>Builds a {@link WorkbenchFrame} on the EDT. Never touches DirectML,
 * FFM, or any other runtime module directly – the only sidecar interaction
 * goes through {@link com.aresstack.windirectml.sidecar.client.SidecarClient}.
 */
public final class WorkbenchApp {

    private WorkbenchApp() {}

    public static void main(String[] args) {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (ClassNotFoundException | InstantiationException
                | IllegalAccessException | UnsupportedLookAndFeelException ignored) {
            // Fall back to default L&F.
        }
        SwingUtilities.invokeLater(new Runnable() {
            @Override public void run() {
                WorkbenchFrame f = new WorkbenchFrame();
                f.setVisible(true);
            }
        });
    }
}

