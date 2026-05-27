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

        // Summarizer / decoder models
        var phi3Btn = new JButton("Download Phi-3 Mini 4K Instruct (Summarizer)");
        phi3Btn.addActionListener(e -> startPhi3Download());
        buttons.add(phi3Btn);

        // Qwen2.5-Coder – download is available but runtime is not yet supported
        var qwenBtn = new JButton("Download Qwen2.5-Coder 0.5B (planned)");
        qwenBtn.setToolTipText("Qwen2.5-Coder 0.5B Instruct – ONNX source TBD/research. "
                + "Runtime not yet available; download only.");
        qwenBtn.addActionListener(e -> startQwenDownload());
        qwenBtn.setEnabled(false); // disabled until ONNX source is verified (#96/#99)
        buttons.add(qwenBtn);

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
        var targetDir = model.getModelRoot().resolve("qwen2.5-coder-0.5b-directml-int4");
        appendLog("Starting Qwen2.5-Coder 0.5B download (ONNX INT4 AWQ block-128 layout from "
                + ModelDownloader.QWEN_SUBDIR + ") -> " + targetDir);
        appendLog("  Required files: " + ModelDownloader.QWEN_REQUIRED_FILES);
        appendLog("  NOTE: ONNX source is TBD/research. Download may fail if source is not yet available.");

        new SwingWorker<Boolean, String>() {
            @Override
            protected Boolean doInBackground() {
                try {
                    // Source repo is TBD – candidate: onnx-community/Qwen2.5-Coder-0.5B-Instruct
                    // This will be updated once source verification completes (#96).
                    ModelDownloader.downloadQwen(
                            "onnx-community/Qwen2.5-Coder-0.5B-Instruct",
                            targetDir, force, this::publish);
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
