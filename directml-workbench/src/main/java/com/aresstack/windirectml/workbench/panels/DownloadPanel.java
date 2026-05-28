package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.workbench.WorkbenchModel;
import com.aresstack.windirectml.workbench.download.DownloadFolderOpener;
import com.aresstack.windirectml.workbench.download.ModelDownloader;
import com.aresstack.windirectml.workbench.download.ModelDownloadUrls;
import com.aresstack.windirectml.workbench.download.QwenModelDownloadConfig;

import javax.swing.*;
import java.awt.*;
import java.awt.datatransfer.StringSelection;
import java.nio.file.Path;
import java.util.List;

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

        // Buttons panel – each row: [Download] [Copy URL] [Open Folder]
        var buttons = new JPanel(new GridLayout(0, 3, 4, 4));

        addEmbeddingRow(buttons, "Download MiniLM (all-MiniLM-L6-v2)",
                "sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2");
        addEmbeddingRow(buttons, "Download E5 small-v2",
                "intfloat/e5-small-v2", "e5-small-v2");
        addEmbeddingRow(buttons, "Download E5 base-v2",
                "intfloat/e5-base-v2", "e5-base-v2");
        addEmbeddingRow(buttons, "Download E5 large-v2",
                "intfloat/e5-large-v2", "e5-large-v2");
        addEmbeddingRow(buttons, "Download Reranker (ms-marco-MiniLM-L-6-v2)",
                "cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder-ms-marco-MiniLM-L-6-v2");
        addEmbeddingRow(buttons, "Download Reranker (ms-marco-MiniLM-L-12-v2)",
                "cross-encoder/ms-marco-MiniLM-L-12-v2", "cross-encoder-ms-marco-MiniLM-L-12-v2");

        // Summarizer / decoder models
        addPhi3Row(buttons);
        addQwenRow(buttons);

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

    // ---- Row builders ----

    private void addEmbeddingRow(JPanel buttons, String label, String repo, String folder) {
        var downloadBtn = new JButton(label);
        downloadBtn.addActionListener(e -> startDownload(repo, folder));
        buttons.add(downloadBtn);

        buttons.add(createCopyUrlButton(() -> ModelDownloadUrls.forEmbeddingModel(repo)));
        buttons.add(createOpenFolderButton(() -> model.getModelRoot().resolve(folder)));
    }

    private void addPhi3Row(JPanel buttons) {
        var phi3Btn = new JButton("Download Phi-3 Mini 4K Instruct (Summarizer)");
        phi3Btn.addActionListener(e -> startPhi3Download());
        buttons.add(phi3Btn);

        buttons.add(createCopyUrlButton(ModelDownloadUrls::forPhi3));
        buttons.add(createOpenFolderButton(() -> model.getModelRoot().resolve("phi-3-mini-4k-instruct-onnx")));
    }

    private void addQwenRow(JPanel buttons) {
        var qwenBtn = new JButton("Download Qwen2.5-Coder 0.5B");
        qwenBtn.setToolTipText("Download Qwen2.5-Coder 0.5B for local Workbench testing.");
        qwenBtn.addActionListener(e -> startQwenDownload());
        buttons.add(qwenBtn);

        var config = QwenModelDownloadConfig.DEFAULT;
        buttons.add(createCopyUrlButton(() -> ModelDownloadUrls.forQwen(config)));
        buttons.add(createOpenFolderButton(() -> model.getModelRoot().resolve(config.localDirName())));
    }

    // ---- Copy URL button ----

    private JButton createCopyUrlButton(java.util.function.Supplier<List<String>> urlsSupplier) {
        var btn = new JButton("\uD83D\uDCCB"); // clipboard emoji as compact icon
        btn.setToolTipText("Copy download URL");
        btn.setMargin(new Insets(2, 4, 2, 4));
        btn.addActionListener(e -> {
            var urls = urlsSupplier.get();
            String text = String.join("\n", urls);
            var selection = new StringSelection(text);
            Toolkit.getDefaultToolkit().getSystemClipboard().setContents(selection, null);
            appendLog("Copied " + urls.size() + " download URL(s) to clipboard.");
        });
        return btn;
    }

    // ---- Open folder button ----

    private JButton createOpenFolderButton(java.util.function.Supplier<Path> folderSupplier) {
        var btn = new JButton("\uD83D\uDCC2"); // open folder emoji as compact icon
        btn.setToolTipText("Open target folder");
        btn.setMargin(new Insets(2, 4, 2, 4));
        btn.addActionListener(e -> {
            Path folder = folderSupplier.get();
            DownloadFolderOpener.openFolder(folder, this::appendLog);
        });
        return btn;
    }

    private void startDownload(String repo, String folder) {
        boolean force = forceCheckbox.isSelected();
        var targetDir = model.getModelRoot().resolve(folder);
        appendLog("Starting download: " + repo + " -> " + targetDir);

        new SwingWorker<Boolean, String>() {
            @Override
            protected Boolean doInBackground() {
                try {
                    ModelDownloader.download(repo, targetDir, force, this::publish);
                    return true;
                } catch (Exception ex) {
                    publish("ERROR: " + ex.getMessage());
                    return false;
                }
            }

            @Override
            protected void process(java.util.List<String> chunks) {
                for (var msg : chunks) appendLog(msg);
            }

            @Override
            protected void done() {
                try {
                    boolean ok = get();
                    appendLog((ok ? "Download finished: " : "Download ended with errors: ") + folder);
                } catch (Exception ex) {
                    appendLog("Download ended with errors: " + folder + " (" + ex.getMessage() + ")");
                }
            }
        }.execute();
    }

    private void startPhi3Download() {
        boolean force = forceCheckbox.isSelected();
        var targetDir = model.getModelRoot().resolve("phi-3-mini-4k-instruct-onnx");
        appendLog("Starting Phi-3 download (ONNX/GenAI layout from "
                + ModelDownloader.PHI3_SUBDIR + ") -> " + targetDir);
        appendLog("  Required files: " + ModelDownloader.PHI3_REQUIRED_FILES);

        new SwingWorker<Boolean, String>() {
            @Override
            protected Boolean doInBackground() {
                try {
                    ModelDownloader.downloadPhi3(targetDir, force, this::publish);
                    return true;
                } catch (Exception ex) {
                    publish("ERROR: " + ex.getMessage());
                    return false;
                }
            }

            @Override
            protected void process(java.util.List<String> chunks) {
                for (var msg : chunks) appendLog(msg);
            }

            @Override
            protected void done() {
                try {
                    boolean ok = get();
                    appendLog((ok ? "Phi-3 download finished: "
                            : "Phi-3 download ended with errors: ") + targetDir);
                } catch (Exception ex) {
                    appendLog("Phi-3 download ended with errors: " + ex.getMessage());
                }
            }
        }.execute();
    }

    private void startQwenDownload() {
        boolean force = forceCheckbox.isSelected();
        var config = com.aresstack.windirectml.workbench.download.QwenModelDownloadConfig.DEFAULT;
        var targetDir = model.getModelRoot().resolve(config.localDirName());
        appendLog("Starting Qwen2.5-Coder 0.5B download -> " + targetDir);
        appendLog("  Required files: " + config.requiredLocalFiles());

        new SwingWorker<Boolean, String>() {
            @Override
            protected Boolean doInBackground() {
                try {
                    ModelDownloader.downloadQwen(config, targetDir, force, this::publish);
                    return true;
                } catch (Exception ex) {
                    publish("ERROR: " + ex.getMessage());
                    return false;
                }
            }

            @Override
            protected void process(java.util.List<String> chunks) {
                for (var msg : chunks) appendLog(msg);
            }

            @Override
            protected void done() {
                try {
                    boolean ok = get();
                    appendLog((ok ? "Qwen download finished: "
                            : "Qwen download ended with errors: ") + targetDir);
                } catch (Exception ex) {
                    appendLog("Qwen download ended with errors: " + ex.getMessage());
                }
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
