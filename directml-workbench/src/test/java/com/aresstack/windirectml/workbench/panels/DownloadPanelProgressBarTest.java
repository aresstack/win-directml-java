package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.workbench.WorkbenchModel;
import org.junit.jupiter.api.Test;

import javax.swing.JProgressBar;
import javax.swing.SwingUtilities;
import java.awt.Component;
import java.awt.Container;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertTrue;

class DownloadPanelProgressBarTest {

    @Test
    void downloadPanelProvidesProgressBarForEachDownloadableModel() throws Exception {
        AtomicReference<DownloadPanel> panelRef = new AtomicReference<>();
        SwingUtilities.invokeAndWait(() -> panelRef.set(new DownloadPanel(new WorkbenchModel())));

        List<JProgressBar> progressBars = new ArrayList<>();
        collectProgressBars(panelRef.get(), progressBars);

        assertTrue(progressBars.size() >= 13,
                "Expected one progress bar for each download row plus selected Qwen, got " + progressBars.size());
        assertTrue(progressBars.stream().allMatch(JProgressBar::isStringPainted),
                "Progress bars should show textual progress state");
        assertTrue(progressBars.stream().allMatch(bar -> bar.getMinimum() == 0 && bar.getMaximum() == 100),
                "Progress bars should use percentage bounds");
    }

    private static void collectProgressBars(Component component, List<JProgressBar> out) {
        if (component instanceof JProgressBar progressBar) {
            out.add(progressBar);
        }
        if (component instanceof Container container) {
            for (Component child : container.getComponents()) {
                collectProgressBars(child, out);
            }
        }
    }
}
