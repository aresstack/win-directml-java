package com.aresstack.windirectml.workbench;

import com.aresstack.windirectml.workbench.panels.*;

import javax.swing.*;
import java.awt.*;

/**
 * Main frame with tabbed panels for the DirectML Workbench.
 */
public final class WorkbenchFrame extends JFrame {

    private final WorkbenchModel model = new WorkbenchModel();

    public WorkbenchFrame() {
        super("DirectML Workbench – Java 21 Direct API");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setSize(900, 700);
        setLocationRelativeTo(null);

        var tabs = new JTabbedPane();
        tabs.addTab("Config", new ConfigPanel(model));
        tabs.addTab("Download", new DownloadPanel(model));
        tabs.addTab("Embeddings", new EmbeddingsPanel(model));
        tabs.addTab("Batch Embeddings", new BatchEmbeddingsPanel(model));
        tabs.addTab("Reranker", new RerankerPanel(model));
        tabs.addTab("About", new AboutPanel());

        getContentPane().add(tabs, BorderLayout.CENTER);
    }
}
