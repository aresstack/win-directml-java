package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.workbench.WorkbenchModel;
import com.aresstack.windirectml.workbench.download.DownloadFolderOpener;
import com.aresstack.windirectml.workbench.download.DownloadOverrideStore;
import com.aresstack.windirectml.workbench.download.ModelDownloadManifest;
import com.aresstack.windirectml.workbench.download.ModelDownloadUrls;
import com.aresstack.windirectml.workbench.download.ModelDownloader;
import com.aresstack.windirectml.workbench.download.QwenModelDownloadConfig;
import com.aresstack.windirectml.workbench.download.QwenOnnxModelVariant;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import javax.swing.SwingWorker;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.HeadlessException;
import java.awt.Insets;
import java.awt.Toolkit;
import java.awt.datatransfer.StringSelection;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Panel with download buttons for supported models from Hugging Face.
 */
public final class DownloadPanel extends JPanel {

    private final WorkbenchModel model;
    private final JTextArea logArea;
    private final JCheckBox forceCheckbox;
    private final DownloadOverrideStore overrideStore;
    private final Map<String, ModelDownloadManifest> manifests = new HashMap<String, ModelDownloadManifest>();

    private JComboBox<QwenOnnxModelVariant> qwenVariantSelector;
    private JTextField qwenSelectedUrlField;
    private JCheckBox downloadAllQwenVariantsCheckbox;

    public DownloadPanel(WorkbenchModel model) {
        this.model = model;
        this.overrideStore = new DownloadOverrideStore(DownloadOverrideStore.defaultStorePath());
        setLayout(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        JPanel buttons = new JPanel(new GridLayout(0, 3, 4, 4));

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

        forceCheckbox = new JCheckBox("Force re-download (overwrite existing)");

        JPanel topPanel = new JPanel(new BorderLayout(8, 8));
        topPanel.add(buttons, BorderLayout.NORTH);
        topPanel.add(createQwenPanel(), BorderLayout.CENTER);
        topPanel.add(forceCheckbox, BorderLayout.SOUTH);
        add(topPanel, BorderLayout.NORTH);

        logArea = new JTextArea(18, 60);
        logArea.setEditable(false);
        logArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        add(new JScrollPane(logArea), BorderLayout.CENTER);
    }

    private void addEmbeddingRow(JPanel buttons, String label, String repo, String folder) {
        ModelDownloadManifest manifest = overrideStore.applyOverrides(
                ModelDownloadUrls.manifestForEmbedding(repo, folder));
        manifests.put(manifest.modelId(), manifest);

        JButton downloadBtn = new JButton(label);
        downloadBtn.addActionListener(e -> startManifestDownload(folder));
        buttons.add(downloadBtn);

        buttons.add(createConfigButton(folder));
        buttons.add(createOpenFolderButton(() -> model.getModelRoot().resolve(folder)));
    }

    private void addPhi3Row(JPanel buttons) {
        ModelDownloadManifest manifest = overrideStore.applyOverrides(ModelDownloadUrls.manifestForPhi3());
        manifests.put(manifest.modelId(), manifest);
        String modelId = manifest.modelId();

        JButton phi3Btn = new JButton("Download Phi-3 Mini 4K Instruct (Summarizer)");
        phi3Btn.addActionListener(e -> startManifestDownload(modelId));
        buttons.add(phi3Btn);

        buttons.add(createConfigButton(modelId));
        buttons.add(createOpenFolderButton(() -> model.getModelRoot().resolve(manifest.localDirName())));
    }

    private JPanel createQwenPanel() {
        JPanel panel = new JPanel(new BorderLayout(4, 4));
        panel.setBorder(BorderFactory.createTitledBorder("Qwen2.5-Coder 0.5B ONNX"));

        JPanel controls = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 2));
        controls.add(new JLabel("Quantisierung:"));

        qwenVariantSelector = new JComboBox<QwenOnnxModelVariant>(QwenOnnxModelVariant.values());
        qwenVariantSelector.setSelectedItem(QwenOnnxModelVariant.fromModelFileName(model.getQwenModelFile()));
        qwenVariantSelector.addActionListener(e -> updateSelectedQwenVariant());
        controls.add(qwenVariantSelector);

        JButton downloadBtn = new JButton("Download selected Qwen");
        downloadBtn.addActionListener(e -> startQwenDownload());
        controls.add(downloadBtn);

        downloadAllQwenVariantsCheckbox = new JCheckBox("alle ONNX-Varianten herunterladen");
        downloadAllQwenVariantsCheckbox.setToolTipText("Nur aktivieren, wenn alle Qwen-ONNX-Dateien vorab geladen werden sollen.");
        controls.add(downloadAllQwenVariantsCheckbox);

        controls.add(createOpenFolderButton(() -> model.getModelRoot().resolve(QwenModelDownloadConfig.LOCAL_DIR_NAME)));
        panel.add(controls, BorderLayout.NORTH);

        JPanel urlPanel = new JPanel(new BorderLayout(4, 4));
        urlPanel.add(new JLabel("Gewählte HF-URL:"), BorderLayout.WEST);
        qwenSelectedUrlField = new JTextField();
        qwenSelectedUrlField.setEditable(false);
        urlPanel.add(qwenSelectedUrlField, BorderLayout.CENTER);
        JButton copyButton = new JButton("Copy");
        copyButton.addActionListener(e -> copySelectedQwenUrl());
        urlPanel.add(copyButton, BorderLayout.EAST);
        panel.add(urlPanel, BorderLayout.SOUTH);

        updateSelectedQwenVariant();
        return panel;
    }

    private void updateSelectedQwenVariant() {
        QwenOnnxModelVariant variant = selectedQwenVariant();
        model.setQwenModelFile(variant.modelFileName());
        QwenModelDownloadConfig config = QwenModelDownloadConfig.forVariant(variant);
        if (qwenSelectedUrlField != null) {
            qwenSelectedUrlField.setText(ModelDownloadUrls.selectedQwenModelUrl(config));
        }
    }

    private QwenOnnxModelVariant selectedQwenVariant() {
        if (qwenVariantSelector == null) {
            return QwenOnnxModelVariant.fromModelFileName(model.getQwenModelFile());
        }
        Object selected = qwenVariantSelector.getSelectedItem();
        if (selected instanceof QwenOnnxModelVariant) {
            return (QwenOnnxModelVariant) selected;
        }
        return QwenOnnxModelVariant.Q4F16;
    }

    private void copySelectedQwenUrl() {
        if (qwenSelectedUrlField == null) {
            return;
        }
        String url = qwenSelectedUrlField.getText();
        StringSelection selection = new StringSelection(url);
        try {
            Toolkit.getDefaultToolkit().getSystemClipboard().setContents(selection, null);
            appendLog("Copied Qwen URL: " + url);
        } catch (HeadlessException | IllegalStateException | SecurityException ex) {
            JOptionPane.showMessageDialog(this,
                    "Could not copy URL to clipboard: " + ex.getMessage(),
                    "Clipboard unavailable",
                    JOptionPane.WARNING_MESSAGE);
        }
    }

    private JButton createConfigButton(String modelId) {
        JButton btn = new JButton("\u2699");
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
        ModelDownloadManifest manifest = manifests.get(modelId);
        if (manifest == null) {
            return;
        }

        DownloadUrlConfigDialog dialog = new DownloadUrlConfigDialog(SwingUtilities.getWindowAncestor(this), manifest);
        dialog.setVisible(true);

        if (dialog.isAccepted()) {
            ModelDownloadManifest updated = manifest.withAllUrls(dialog.getEditedUrls());
            manifests.put(modelId, updated);
            overrideStore.storeOverrides(updated);
            appendLog("Updated download URLs for: " + modelId);
        }
    }

    private JButton createOpenFolderButton(Supplier<Path> folderSupplier) {
        JButton btn = new JButton("\uD83D\uDCC2");
        btn.setToolTipText("Open target folder");
        btn.setMargin(new Insets(2, 4, 2, 4));
        btn.getAccessibleContext().setAccessibleName("Open target folder");
        btn.addActionListener(e -> {
            Path folder = folderSupplier.get();
            DownloadFolderOpener.openFolder(folder, this::appendLog);
        });
        return btn;
    }

    private void startManifestDownload(String modelId) {
        ModelDownloadManifest manifest = manifests.get(modelId);
        if (manifest == null) {
            return;
        }
        startDownload(manifest, modelId);
    }

    private void startQwenDownload() {
        QwenOnnxModelVariant variant = selectedQwenVariant();
        model.setQwenModelFile(variant.modelFileName());

        ModelDownloadManifest manifest;
        String label;
        if (downloadAllQwenVariantsCheckbox != null && downloadAllQwenVariantsCheckbox.isSelected()) {
            manifest = ModelDownloadUrls.manifestForAllQwenVariants();
            label = manifest.modelId() + " (all ONNX variants)";
            appendLog("Selected Qwen file for runtime remains: " + variant.modelFileName());
        } else {
            QwenModelDownloadConfig config = QwenModelDownloadConfig.forVariant(variant);
            manifest = ModelDownloadUrls.manifestForQwen(config);
            label = manifest.modelId() + " (" + variant.modelFileName() + ")";
            appendLog("Selected Qwen URL: " + ModelDownloadUrls.selectedQwenModelUrl(config));
        }
        startDownload(manifest, label);
    }

    private void startDownload(ModelDownloadManifest manifest, String label) {
        boolean force = forceCheckbox.isSelected();
        Path targetDir = model.getModelRoot().resolve(manifest.localDirName());
        appendLog("Starting download: " + label + " -> " + targetDir);

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
                for (String msg : chunks) {
                    appendLog(msg);
                }
            }

            @Override
            protected void done() {
                try {
                    boolean ok = get();
                    appendLog((ok ? "Download finished: " : "Download ended with errors: ") + label);
                } catch (Exception ex) {
                    appendLog("Download ended with errors: " + label + " (" + ex.getMessage() + ")");
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
