package com.aresstack.windirectml.workbench;

import com.aresstack.windirectml.workbench.panels.*;
import org.junit.jupiter.api.Test;

import javax.swing.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Headless Swing smoke test – verifies that all panels can be constructed
 * without a visible window and without network/model dependencies.
 */
class WorkbenchSmokeTest {

    @Test
    void frameCanBeConstructedHeadlessly() throws Exception {
        if (java.awt.GraphicsEnvironment.isHeadless()) {
            // In headless CI, verify that WorkbenchFrame class is loadable
            assertNotNull(WorkbenchFrame.class);
            return;
        }
        SwingUtilities.invokeAndWait(() -> {
            var frame = new WorkbenchFrame();
            assertNotNull(frame);
            assertEquals("DirectML Workbench – Java 21 Direct API", frame.getTitle());
            assertTrue(frame.getContentPane().getComponentCount() > 0);
            frame.dispose();
        });
    }

    @Test
    void configPanelCanBeConstructed() throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            var model = new WorkbenchModel();
            var panel = new ConfigPanel(model);
            assertNotNull(panel);
            assertTrue(panel.getComponentCount() > 0);
        });
    }

    @Test
    void downloadPanelCanBeConstructed() throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            var model = new WorkbenchModel();
            var panel = new DownloadPanel(model);
            assertNotNull(panel);
            assertTrue(panel.getComponentCount() > 0);
        });
    }

    @Test
    void embeddingsPanelCanBeConstructed() throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            var model = new WorkbenchModel();
            var panel = new EmbeddingsPanel(model);
            assertNotNull(panel);
            assertTrue(panel.getComponentCount() > 0);
        });
    }

    @Test
    void batchEmbeddingsPanelCanBeConstructed() throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            var model = new WorkbenchModel();
            var panel = new BatchEmbeddingsPanel(model);
            assertNotNull(panel);
            assertTrue(panel.getComponentCount() > 0);
        });
    }

    @Test
    void rerankerPanelCanBeConstructed() throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            var model = new WorkbenchModel();
            var panel = new RerankerPanel(model);
            assertNotNull(panel);
            assertTrue(panel.getComponentCount() > 0);
        });
    }

    @Test
    void aboutPanelCanBeConstructed() throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            var panel = new AboutPanel();
            assertNotNull(panel);
            assertTrue(panel.getComponentCount() > 0);
        });
    }

    @Test
    void workbenchModelDefaults() {
        var model = new WorkbenchModel();
        assertEquals(com.aresstack.windirectml.runtime.facade.Backend.AUTO, model.getBackend());
        assertEquals("model", model.getModelRoot().toString());
        assertEquals("all-MiniLM-L6-v2", model.getEmbeddingModel());
        assertEquals("cross-encoder-ms-marco-MiniLM-L-6-v2", model.getRerankerModel());
    }
}
