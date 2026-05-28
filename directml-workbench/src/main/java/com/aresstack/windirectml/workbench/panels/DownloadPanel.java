package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.workbench.WorkbenchModel;
import com.aresstack.windirectml.workbench.download.*;

import javax.swing.*;
import java.awt.*;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * Panel with download buttons for supported models from Hugging Face.
 */
public final class DownloadPanel extends JPanel {

    private final WorkbenchModel model;
    private final JTextArea logArea;
    private final JCheckBox forceCheckbox;
    private final DownloadOverrideStore overrideStore;

    /** In-memory manifests keyed by model id, updated when user presses OK in config dialog. */
    private final Map<String, ModelDownloadManifest> manifests = new HashMap<>();

    public DownloadPanel(WorkbenchModel model) {
        this.model = model;
        this.overrideStore = new DownloadOverrideStore(DownloadOverrideStore.defaultStorePath());
        setLayout(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        // Buttons panel – each row: [Download] [Config] [Open Folder]
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
        var manifest = overrideStore.applyOverrides(
                ModelDownloadUrls.manifestForEmbedding(repo, folder));
        manifests.put(manifest.modelId(), manifest);

        var downloadBtn = new JButton(label);
        downloadBtn.addActionListener(e -> startManifestDownload(folder));
        buttons.add(downloadBtn);

        buttons.add(createConfigButton(folder));
        buttons.add(createOpenFolderButton(() -> model.getModelRoot().resolve(folder)));
    }

    private void addPhi3Row(JPanel buttons) {
        var manifest = overrideStore.applyOverrides(ModelDownloadUrls.manifestForPhi3());
        manifests.put(manifest.modelId(), manifest);
        String modelId = manifest.modelId();

        var phi3Btn = new JButton("Download Phi-3 Mini 4K Instruct (Summarizer)");
        phi3Btn.addActionListener(e -> startManifestDownload(modelId));
        buttons.add(phi3Btn);

        buttons.add(createConfigButton(modelId));
        buttons.add(createOpenFolderButton(() -> model.getModelRoot().resolve(manifest.localDirName())));
    }

    private void addQwenRow(JPanel buttons) {
        var config = QwenModelDownloadConfig.DEFAULT;
        var manifest = overrideStore.applyOverrides(ModelDownloadUrls.manifestForQwen(config));
        manifests.put(manifest.modelId(), manifest);
        String modelId = manifest.modelId();

        var qwenBtn = new JButton("Download Qwen2.5-Coder 0.5B");
        qwenBtn.setToolTipText("Download Qwen2.5-Coder 0.5B for local Workbench testing.");
        qwenBtn.addActionListener(e -> startManifestDownload(modelId));
        buttons.add(qwenBtn);

        buttons.add(createConfigButton(modelId));
        buttons.add(createOpenFolderButton(() -> model.getModelRoot().resolve(config.localDirName())));
    }

    // ---- Config button ----

    private JButton createConfigButton(String modelId) {
        var btn = new JButton("\u2699"); // gear icon
        btn.setToolTipText("Configure download URLs");
        btn.setMargin(new Insets(0, 0, 0, 0));
        Dimension sq = new Dimension(28, 28);
        btn.setPreferredSize(sq);
        btn.setMinimumSize(sq);
        btn.setMaximumSize(sq);
        btn.getAccessibleContext().setAccessibleName("Configure download URLs");
        btn.addActionListener(e -> openConfigDialog(modelId));
        return btn;
    }

    private void openConfigDialog(String modelId) {
        var manifest = manifests.get(modelId);
        if (manifest == null) return;

        var dialog = new DownloadUrlConfigDialog(
                SwingUtilities.getWindowAncestor(this), manifest);
        dialog.setVisible(true); // blocks (modal)

        if (dialog.isAccepted()) {
            var updated = manifest.withAllUrls(dialog.getEditedUrls());
            manifests.put(modelId, updated);
            overrideStore.storeOverrides(updated);
            appendLog("Updated download URLs for: " + modelId);
        }
    }

    // ---- Open folder button ----

    private JButton createOpenFolderButton(java.util.function.Supplier<Path> folderSupplier) {
        var btn = new JButton("\uD83D\uDCC2"); // open folder emoji as compact icon
        btn.setToolTipText("Open target folder");
        btn.setMargin(new Insets(2, 4, 2, 4));
        btn.getAccessibleContext().setAccessibleName("Open target folder");
        btn.addActionListener(e -> {
            Path folder = folderSupplier.get();
            DownloadFolderOpener.openFolder(folder, this::appendLog);
        });
        return btn;
    }

    // ---- Download using manifest ----

    private void startManifestDownload(String modelId) {
        var manifest = manifests.get(modelId);
        if (manifest == null) return;

        boolean force = forceCheckbox.isSelected();
        var targetDir = model.getModelRoot().resolve(manifest.localDirName());
        appendLog("Starting download: " + modelId + " -> " + targetDir);

        new SwingWorker<Boolean, String>() {
            @Override
            protected Boolean doInBackground() {
                try {
                    ModelDownloader.downloadFromManifest(manifest, targetDir, force, this::publish);
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
                    appendLog((ok ? "Download finished: " : "Download ended with errors: ") + modelId);
                } catch (Exception ex) {
                    appendLog("Download ended with errors: " + modelId + " (" + ex.getMessage() + ")");
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
