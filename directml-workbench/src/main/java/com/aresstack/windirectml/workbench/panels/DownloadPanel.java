package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.workbench.WorkbenchModel;
import com.aresstack.windirectml.workbench.download.ModelDownloader;

import javax.swing.*;
import java.awt.*;

/**
 * Panel with download buttons for supported models from Hugging Face.
 */
public final class DownloadPanel extends JPanel {

    private final WorkbenchModel model;
    private final JTextArea logArea;
    private final JCheckBox forceCheckbox;

    public DownloadPanel(WorkbenchModel model) {
        this.model = model;
        setLayout(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        // Buttons panel
        var buttons = new JPanel(new GridLayout(0, 2, 8, 8));

        buttons.add(createDownloadButton("Download MiniLM (all-MiniLM-L6-v2)",
                "sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2"));
        buttons.add(createDownloadButton("Download E5 small-v2",
                "intfloat/e5-small-v2", "e5-small-v2"));
        buttons.add(createDownloadButton("Download E5 base-v2",
                "intfloat/e5-base-v2", "e5-base-v2"));
        buttons.add(createDownloadButton("Download E5 large-v2",
                "intfloat/e5-large-v2", "e5-large-v2"));
        buttons.add(createDownloadButton("Download Reranker (ms-marco-MiniLM-L-6-v2)",
                "cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder-ms-marco-MiniLM-L-6-v2"));
        buttons.add(createDownloadButton("Download Reranker (ms-marco-MiniLM-L-12-v2)",
                "cross-encoder/ms-marco-MiniLM-L-12-v2", "cross-encoder-ms-marco-MiniLM-L-12-v2"));

        forceCheckbox = new JCheckBox("Force re-download (overwrite existing)");

        var topPanel = new JPanel(new BorderLayout(8, 8));
        topPanel.add(buttons, BorderLayout.CENTER);
        topPanel.add(forceCheckbox, BorderLayout.SOUTH);
        add(topPanel, BorderLayout.NORTH);

        // Log area
        logArea = new JTextArea(18, 60);
        logArea.setEditable(false);
        logArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        add(new JScrollPane(logArea), BorderLayout.CENTER);
    }

    private JButton createDownloadButton(String label, String repo, String folder) {
        var btn = new JButton(label);
        btn.addActionListener(e -> startDownload(repo, folder));
        return btn;
    }

    private void startDownload(String repo, String folder) {
        boolean force = forceCheckbox.isSelected();
        var targetDir = model.getModelRoot().resolve(folder);
        appendLog("Starting download: " + repo + " -> " + targetDir);

        new SwingWorker<Void, String>() {
            @Override
            protected Void doInBackground() {
                try {
                    ModelDownloader.download(repo, targetDir, force, this::publish);
                } catch (Exception ex) {
                    publish("ERROR: " + ex.getMessage());
                }
                return null;
            }

            @Override
            protected void process(java.util.List<String> chunks) {
                for (var msg : chunks) appendLog(msg);
            }

            @Override
            protected void done() {
                appendLog("Download complete: " + folder);
            }
        }.execute();
    }

    private void appendLog(String message) {
        SwingUtilities.invokeLater(() -> {
            logArea.append(message + "\n");
            logArea.setCaretPosition(logArea.getDocument().getLength());
        });
    }
}
