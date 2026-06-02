package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.inference.qwen.QwenModelDirValidator;
import com.aresstack.windirectml.workbench.WorkbenchModel;
import com.aresstack.windirectml.workbench.download.*;

import javax.swing.*;
import java.awt.*;
import java.awt.datatransfer.StringSelection;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * Panel with download controls for supported models from Hugging Face.
 */
public final class DownloadPanel extends JPanel {

    private static final String QWEN_MODEL_ID = QwenOnnxModelVariant.LOCAL_DIR_NAME;

    private final WorkbenchModel model;
    private final JTextArea logArea;
    private final JCheckBox forceCheckbox;
    private final DownloadOverrideStore overrideStore;

    private JComboBox<QwenOnnxModelVariant> qwenVariantCombo;
    private JTextField qwenUrlField;
    private JCheckBox qwenDownloadAllCheckbox;

    /**
     * In-memory manifests keyed by model id, updated when user presses OK in config dialog.
     */
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

        addPhi3Row(buttons);

        var downloadsPanel = new JPanel(new BorderLayout(8, 8));
        downloadsPanel.add(buttons, BorderLayout.NORTH);
        downloadsPanel.add(createQwenVariantPanel(), BorderLayout.CENTER);

        forceCheckbox = new JCheckBox("Force re-download (overwrite existing)");

        var topPanel = new JPanel(new BorderLayout(8, 8));
        topPanel.add(downloadsPanel, BorderLayout.CENTER);
        topPanel.add(forceCheckbox, BorderLayout.SOUTH);
        add(topPanel, BorderLayout.NORTH);

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

    private JPanel createQwenVariantPanel() {
        var panel = new JPanel(new GridBagLayout());
        panel.setBorder(BorderFactory.createTitledBorder("Qwen2.5-Coder 0.5B ONNX variant"));

        var gbc = new GridBagConstraints();
        gbc.insets = new Insets(3, 3, 3, 3);
        gbc.anchor = GridBagConstraints.WEST;
        gbc.fill = GridBagConstraints.HORIZONTAL;

        QwenOnnxModelVariant activeVariant = readActiveQwenVariant();
        qwenVariantCombo = new JComboBox<>(QwenOnnxModelVariant.orderedValues()
                .toArray(new QwenOnnxModelVariant[0]));
        qwenVariantCombo.setSelectedItem(activeVariant);
        qwenVariantCombo.setToolTipText("Choose the Hugging Face ONNX filename that the Qwen runtime should use.");
        qwenVariantCombo.addActionListener(e -> updateQwenSelection(true));

        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.weightx = 0;
        panel.add(new JLabel("ONNX file:"), gbc);

        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(qwenVariantCombo, gbc);

        qwenDownloadAllCheckbox = new JCheckBox("Download all ONNX files");
        qwenDownloadAllCheckbox.setToolTipText("Download every listed ONNX variant. Leave off to download only the selected file.");
        qwenDownloadAllCheckbox.addActionListener(e -> updateQwenManifest());
        gbc.gridx = 2;
        gbc.weightx = 0;
        panel.add(qwenDownloadAllCheckbox, gbc);

        qwenUrlField = new JTextField(ModelDownloadUrls.qwenVariantUrl(activeVariant), 70);
        qwenUrlField.setEditable(false);
        qwenUrlField.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));

        gbc.gridx = 0;
        gbc.gridy = 1;
        gbc.weightx = 0;
        panel.add(new JLabel("Selected URL:"), gbc);

        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(qwenUrlField, gbc);

        var copyBtn = new JButton("Copy URL");
        copyBtn.addActionListener(e -> copyTextToClipboard(qwenUrlField.getText()));
        gbc.gridx = 2;
        gbc.weightx = 0;
        panel.add(copyBtn, gbc);

        var downloadBtn = new JButton("Download Qwen");
        downloadBtn.setToolTipText("Download the selected Qwen ONNX file and support files.");
        downloadBtn.addActionListener(e -> startQwenDownload());
        gbc.gridx = 0;
        gbc.gridy = 2;
        gbc.weightx = 0;
        panel.add(downloadBtn, gbc);

        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(createConfigButton(QWEN_MODEL_ID), gbc);

        gbc.gridx = 2;
        gbc.weightx = 0;
        panel.add(createOpenFolderButton(this::qwenModelDir), gbc);

        updateQwenManifest();
        return panel;
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
        if (QWEN_MODEL_ID.equals(modelId)) {
            updateQwenManifest();
        }
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

    private void startQwenDownload() {
        QwenOnnxModelVariant selectedVariant = selectedQwenVariant();
        updateQwenManifest();
        ModelDownloadManifest manifest = manifests.get(QWEN_MODEL_ID);
        if (manifest == null) {
            appendLog("ERROR: Qwen download manifest is missing.");
            return;
        }

        boolean force = forceCheckbox.isSelected();
        Path targetDir = qwenModelDir();
        appendLog("Selected Qwen ONNX: " + selectedVariant.fileName());
        appendLog("Selected Qwen URL: " + ModelDownloadUrls.qwenVariantUrl(selectedVariant));
        appendLog("Starting download: " + QWEN_MODEL_ID + " -> " + targetDir
                + (qwenDownloadAllCheckbox.isSelected() ? " (all ONNX files)" : ""));

        new SwingWorker<Boolean, String>() {
            @Override
            protected Boolean doInBackground() {
                try {
                    ModelDownloader.downloadFromManifest(manifest, targetDir, force, this::publish);
                    QwenModelDirValidator.writeSelectedOnnxFilename(targetDir, selectedVariant.fileName());
                    publish("  Active Qwen ONNX file: " + selectedVariant.fileName());
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
                    appendLog(ok ? "Download finished: " + QWEN_MODEL_ID
                            : "Download ended with errors: " + QWEN_MODEL_ID);
                } catch (Exception ex) {
                    appendLog("Download ended with errors: " + QWEN_MODEL_ID + " (" + ex.getMessage() + ")");
                }
            }
        }.execute();
    }

    private void updateQwenSelection(boolean persistSelection) {
        QwenOnnxModelVariant selectedVariant = selectedQwenVariant();
        if (qwenUrlField != null) {
            qwenUrlField.setText(ModelDownloadUrls.qwenVariantUrl(selectedVariant));
        }
        updateQwenManifest();
        if (persistSelection) {
            try {
                QwenModelDirValidator.writeSelectedOnnxFilename(qwenModelDir(), selectedVariant.fileName());
                appendLog("Active Qwen ONNX file: " + selectedVariant.fileName());
                appendLog("Selected Qwen URL: " + ModelDownloadUrls.qwenVariantUrl(selectedVariant));
            } catch (IOException ex) {
                appendLog("ERROR: Could not store Qwen ONNX selection: " + ex.getMessage());
            }
        }
    }

    private void updateQwenManifest() {
        QwenOnnxModelVariant selectedVariant = selectedQwenVariant();
        ModelDownloadManifest manifest = qwenDownloadAllCheckbox != null && qwenDownloadAllCheckbox.isSelected()
                ? ModelDownloadUrls.manifestForAllQwenVariants()
                : ModelDownloadUrls.manifestForQwenVariant(selectedVariant);
        manifests.put(QWEN_MODEL_ID, overrideStore.applyOverrides(manifest));
    }

    private QwenOnnxModelVariant selectedQwenVariant() {
        if (qwenVariantCombo == null || qwenVariantCombo.getSelectedItem() == null) {
            return QwenOnnxModelVariant.Q4F16;
        }
        return (QwenOnnxModelVariant) qwenVariantCombo.getSelectedItem();
    }

    private QwenOnnxModelVariant readActiveQwenVariant() {
        Path selectionFile = qwenModelDir().resolve(QwenModelDirValidator.SELECTED_ONNX_FILE);
        if (!Files.exists(selectionFile)) {
            return QwenOnnxModelVariant.Q4F16;
        }
        return QwenOnnxModelVariant.fromFileName(
                QwenModelDirValidator.selectedOnnxFilename(qwenModelDir()));
    }

    private Path qwenModelDir() {
        return model.getModelRoot().resolve(QwenOnnxModelVariant.LOCAL_DIR_NAME);
    }

    private void copyTextToClipboard(String text) {
        var selection = new StringSelection(text);
        try {
            Toolkit.getDefaultToolkit().getSystemClipboard().setContents(selection, null);
            appendLog("Copied Qwen URL to clipboard.");
        } catch (HeadlessException | IllegalStateException | SecurityException ex) {
            appendLog("ERROR: Could not copy URL to clipboard: " + ex.getMessage());
        }
    }

    private void appendLog(String message) {
        SwingUtilities.invokeLater(() -> {
            logArea.append(message + "\n");
            logArea.setCaretPosition(logArea.getDocument().getLength());
        });
    }
}
