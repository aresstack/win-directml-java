package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.inference.artifact.CompilerMissingLifecycle;
import com.aresstack.windirectml.inference.artifact.ModelConversionResult;
import com.aresstack.windirectml.inference.artifact.ModelFamily;
import com.aresstack.windirectml.inference.artifact.ModelPackageLifecycle;
import com.aresstack.windirectml.inference.artifact.QwenPackageLifecycle;
import com.aresstack.windirectml.inference.artifact.SmolLM2PackageLifecycle;
import com.aresstack.windirectml.inference.artifact.T5PackageLifecycle;
import com.aresstack.windirectml.workbench.WorkbenchModel;
import com.aresstack.windirectml.workbench.artifact.ModelArtifactRow;
import com.aresstack.windirectml.workbench.artifact.ModelRuntimeRegistry;
import com.aresstack.windirectml.workbench.download.DownloadAccessSettings;
import com.aresstack.windirectml.workbench.download.DownloadFolderOpener;
import com.aresstack.windirectml.workbench.download.DownloadOverrideStore;
import com.aresstack.windirectml.workbench.download.DownloadUrlOpener;
import com.aresstack.windirectml.workbench.download.ModelDownloadManifest;
import com.aresstack.windirectml.workbench.download.ModelDownloadUrls;
import com.aresstack.windirectml.workbench.download.ModelDownloader;
import com.aresstack.windirectml.workbench.download.ModelFileDescriptor;
import com.aresstack.windirectml.workbench.download.QwenDownloadSource;
import com.aresstack.windirectml.workbench.download.QwenModelDownloadConfig;
import com.aresstack.windirectml.workbench.download.QwenOnnxModelVariant;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import javax.swing.SwingWorker;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.HeadlessException;
import java.awt.Insets;
import java.awt.Toolkit;
import java.awt.Window;
import java.awt.datatransfer.StringSelection;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Panel with download buttons for supported models from Hugging Face.
 */
public final class DownloadPanel extends JPanel {

    private static final int ICON_BUTTON_SIZE = 28;
    private static final String SETTINGS_BUTTON_ICON = "\uD83D\uDEE0";
    private static final Insets ICON_BUTTON_MARGIN = new Insets(0, 0, 0, 0);

    private final WorkbenchModel model;
    private final JTextArea logArea;
    private final JCheckBox forceCheckbox;
    private final DownloadOverrideStore overrideStore;
    private final Map<String, ModelDownloadManifest> manifests = new HashMap<String, ModelDownloadManifest>();
    private final Map<String, JProgressBar> downloadProgressBars = new HashMap<String, JProgressBar>();
    private final List<RowControls> rowControls = new ArrayList<RowControls>();
    private final ModelRuntimeRegistry runtimeRegistry;

    private int downloadRowIndex;
    private boolean downloadAllQwenVariants;
    private JProgressBar qwenDownloadProgressBar;

    public DownloadPanel(WorkbenchModel model) {
        this.model = model;
        this.runtimeRegistry = new ModelRuntimeRegistry(model);
        this.overrideStore = new DownloadOverrideStore(DownloadOverrideStore.defaultStorePath());
        setLayout(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        JPanel downloadRows = new JPanel(new GridBagLayout());
        downloadRowIndex = 0;

        addEmbeddingRow(downloadRows, "Download MiniLM (all-MiniLM-L6-v2)",
                "sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2", ModelFamily.EMBEDDING);
        addEmbeddingRow(downloadRows, "Download E5 small-v2",
                "intfloat/e5-small-v2", "e5-small-v2", ModelFamily.EMBEDDING);
        addEmbeddingRow(downloadRows, "Download E5 base-v2",
                "intfloat/e5-base-v2", "e5-base-v2", ModelFamily.EMBEDDING);
        addEmbeddingRow(downloadRows, "Download E5 large-v2",
                "intfloat/e5-large-v2", "e5-large-v2", ModelFamily.EMBEDDING);
        addEmbeddingRow(downloadRows, "Download Reranker (ms-marco-MiniLM-L-6-v2)",
                "cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder-ms-marco-MiniLM-L-6-v2", ModelFamily.RERANKER);
        addEmbeddingRow(downloadRows, "Download Reranker (ms-marco-MiniLM-L-12-v2)",
                "cross-encoder/ms-marco-MiniLM-L-12-v2", "cross-encoder-ms-marco-MiniLM-L-12-v2", ModelFamily.RERANKER);
        addPhi3Row(downloadRows);
        addQwenRow(downloadRows);
        addManifestRow(downloadRows, "Download Qwen2.5-Coder 1.5B Instruct (SafeTensors, planned)",
                ModelDownloadUrls.manifestForQwenCoder1_5BSafeTensors(), ModelFamily.QWEN);
        addManifestRow(downloadRows, "Download Qwen2.5-Coder 3B Instruct (SafeTensors, planned)",
                ModelDownloadUrls.manifestForQwenCoder3BSafeTensors(), ModelFamily.QWEN);
        addManifestRow(downloadRows, "Download SmolLM2 135M Instruct (Summarizer planned)",
                ModelDownloadUrls.manifestForSmolLm2_135M(), ModelFamily.SMOLLM2);
        addManifestRow(downloadRows, "Download SmolLM2 360M Instruct (Summarizer planned)",
                ModelDownloadUrls.manifestForSmolLm2_360M(), ModelFamily.SMOLLM2);
        addGemmaRow(downloadRows, "Download Gemma 3 270M (gated, planned)",
                ModelDownloadUrls.manifestForGemma3_270M());
        addGemmaRow(downloadRows, "Download Gemma 3 270M Instruct (gated, planned)",
                ModelDownloadUrls.manifestForGemma3_270MInstruct());
        addT5Row(downloadRows, "Download google/flan-t5-small (SafeTensors)",
                ModelDownloadUrls.manifestForGoogleFlanT5Small());
        addT5Row(downloadRows, "Download google-t5/t5-small (SafeTensors smoke-test)",
                ModelDownloadUrls.manifestForGoogleT5Small());
        addCodeT5Row(downloadRows);

        forceCheckbox = new JCheckBox("Force re-download (overwrite existing)");

        JButton refreshStatusButton = new JButton("Check packages / Refresh status");
        refreshStatusButton.setToolTipText("Inspect raw + package status for every model. Never writes or compiles.");
        refreshStatusButton.addActionListener(e -> refreshAllRows());
        JPanel actionsRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 0));
        actionsRow.add(refreshStatusButton);
        actionsRow.add(forceCheckbox);

        JPanel topPanel = new JPanel(new BorderLayout(8, 8));
        topPanel.add(downloadRows, BorderLayout.NORTH);
        topPanel.add(actionsRow, BorderLayout.SOUTH);
        add(topPanel, BorderLayout.NORTH);

        logArea = new JTextArea(18, 60);
        logArea.setEditable(false);
        logArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        add(new JScrollPane(logArea), BorderLayout.CENTER);
    }

    private void addEmbeddingRow(JPanel rows, String label, String repo, String folder, ModelFamily family) {
        ModelDownloadManifest manifest = overrideStore.applyOverrides(
                ModelDownloadUrls.manifestForEmbedding(repo, folder));
        manifests.put(manifest.modelId(), manifest);

        JButton downloadButton = new JButton(label);
        downloadButton.addActionListener(e -> startManifestDownload(folder));

        addDownloadRow(rows,
                downloadButton,
                createConfigButton(folder),
                new ModelArtifactRow(family, () -> model.getModelRoot().resolve(folder),
                        () -> family == ModelFamily.RERANKER
                                ? com.aresstack.windirectml.encoder.pack.EncoderPackageLifecycle.reranker()
                                : com.aresstack.windirectml.encoder.pack.EncoderPackageLifecycle.embedding()),
                createOpenFolderButton(() -> model.getModelRoot().resolve(folder)),
                registerProgressBar(manifest));
    }

    private void addPhi3Row(JPanel rows) {
        ModelDownloadManifest manifest = overrideStore.applyOverrides(ModelDownloadUrls.manifestForPhi3());
        manifests.put(manifest.modelId(), manifest);
        String modelId = manifest.modelId();

        JButton downloadButton = new JButton("Download Phi-3 Mini 4K Instruct (Summarizer)");
        downloadButton.addActionListener(e -> startManifestDownload(modelId));

        addDownloadRow(rows,
                downloadButton,
                createConfigButton(modelId),
                new ModelArtifactRow(ModelFamily.PHI3,
                        () -> model.getModelRoot().resolve(manifest.localDirName()),
                        () -> new CompilerMissingLifecycle(ModelFamily.PHI3, java.util.List.of("config.json"),
                                java.util.List.of(java.util.List.of("*.onnx", "model.safetensors")))),
                createOpenFolderButton(() -> model.getModelRoot().resolve(manifest.localDirName())),
                registerProgressBar(manifest));
    }

    private void addQwenRow(JPanel rows) {
        ModelDownloadManifest onnxManifest = overrideStore.applyOverrides(
                ModelDownloadUrls.manifestForQwen(QwenModelDownloadConfig.forVariant(selectedQwenVariant())));
        ModelDownloadManifest safeTensorsManifest = overrideStore.applyOverrides(
                ModelDownloadUrls.manifestForQwenSafeTensors());
        manifests.put(onnxManifest.modelId(), onnxManifest);
        manifests.put(safeTensorsManifest.modelId(), safeTensorsManifest);

        JButton downloadButton = new JButton("Download Qwen2.5-Coder 0.5B Instruct");
        downloadButton.addActionListener(e -> startQwenRuntimeDownload());

        qwenDownloadProgressBar = createDownloadProgressBar();
        downloadProgressBars.put(onnxManifest.modelId(), qwenDownloadProgressBar);
        downloadProgressBars.put(safeTensorsManifest.modelId(), qwenDownloadProgressBar);

        // Qwen 0.5B unified path: the row, convert and runtime all resolve the q4f16 package in the
        // directml-int4 directory via the shared descriptor (model_q4f16.onnx -> model_q4f16.wdmlpack).
        ModelArtifactRow qwenRow = runtimeRegistry.qwen05b().toRow();

        addDownloadRow(rows,
                downloadButton,
                createQwenConfigButton(),
                qwenRow,
                createOpenFolderButton(runtimeRegistry::qwen05bRuntimeDir),
                qwenDownloadProgressBar);
    }

    private void addManifestRow(JPanel rows, String label, ModelDownloadManifest manifest, ModelFamily family) {
        ModelDownloadManifest effectiveManifest = overrideStore.applyOverrides(manifest);
        manifests.put(effectiveManifest.modelId(), effectiveManifest);
        String modelId = effectiveManifest.modelId();

        JButton downloadButton = new JButton(label);
        downloadButton.addActionListener(e -> startManifestDownload(modelId));

        addDownloadRow(rows,
                downloadButton,
                createConfigButton(modelId),
                new ModelArtifactRow(family,
                        () -> model.getModelRoot().resolve(effectiveManifest.localDirName()),
                        lifecycleSupplier(family)),
                createOpenFolderButton(() -> model.getModelRoot().resolve(effectiveManifest.localDirName())),
                registerProgressBar(effectiveManifest));
    }

    private void addCodeT5Row(JPanel rows) {
        addT5Row(rows, "Download CodeT5 small checkpoint", ModelDownloadUrls.manifestForCodeT5Small());
        addT5Row(rows, "Download CodeT5 base multi-sum checkpoint", ModelDownloadUrls.manifestForCodeT5BaseMultiSum());
    }

    private void addGemmaRow(JPanel rows, String downloadLabel, ModelDownloadManifest baseManifest) {
        ModelDownloadManifest manifest = overrideStore.applyOverrides(baseManifest);
        manifests.put(manifest.modelId(), manifest);
        String modelId = manifest.modelId();

        JButton downloadButton = new JButton(downloadLabel);
        downloadButton.setToolTipText("Downloads Gemma files from a gated Hugging Face repository. Configure a token before downloading.");
        downloadButton.addActionListener(e -> startManifestDownload(modelId));

        addDownloadRow(rows,
                downloadButton,
                createAccessConfigButton(modelId),
                new ModelArtifactRow(ModelFamily.GEMMA3,
                        () -> model.getModelRoot().resolve(manifest.localDirName()),
                        () -> new CompilerMissingLifecycle(ModelFamily.GEMMA3,
                                java.util.List.of("model.safetensors", "config.json", "tokenizer.json"),
                                java.util.List.of(java.util.List.of("*.safetensors")))),
                createOpenFolderButton(() -> model.getModelRoot().resolve(manifest.localDirName())),
                registerProgressBar(manifest));
    }

    private void addT5Row(JPanel rows, String downloadLabel, ModelDownloadManifest baseManifest) {
        ModelDownloadManifest manifest = overrideStore.applyOverrides(baseManifest);
        manifests.put(manifest.modelId(), manifest);
        String modelId = manifest.modelId();

        JButton downloadButton = new JButton(downloadLabel);
        downloadButton.setToolTipText("Downloads T5 config/tokenizer files and model weights when the upstream repository provides a supported source.");
        downloadButton.addActionListener(e -> startManifestDownload(modelId));

        addDownloadRow(rows,
                downloadButton,
                createConfigButton(modelId),
                new ModelArtifactRow(ModelFamily.T5,
                        () -> selectedT5TargetDir(manifest.localDirName()),
                        T5PackageLifecycle::new),
                createOpenFolderButton(() -> selectedT5TargetDir(manifest.localDirName())),
                registerProgressBar(manifest));
    }

    /** Supplier of the central lifecycle for a family (variant-fixed; Qwen rows build their own). */
    private static java.util.function.Supplier<ModelPackageLifecycle> lifecycleSupplier(ModelFamily family) {
        return switch (family) {
            case SMOLLM2 -> SmolLM2PackageLifecycle::new;
            case GEMMA3 -> () -> new CompilerMissingLifecycle(ModelFamily.GEMMA3,
                    java.util.List.of("model.safetensors", "config.json", "tokenizer.json"),
                    java.util.List.of(java.util.List.of("*.safetensors")));
            case T5 -> T5PackageLifecycle::new;
            case QWEN -> QwenPackageLifecycle::new; // default model.onnx -> model.wdmlpack
            default -> () -> new CompilerMissingLifecycle(family,
                    java.util.List.of("config.json", "tokenizer.json"),
                    java.util.List.of(java.util.List.of("*.safetensors", "pytorch_model.bin")));
        };
    }

    private void addDownloadRow(JPanel rows,
                                JButton downloadButton,
                                JButton configButton,
                                ModelArtifactRow artifactRow,
                                JButton openFolderButton,
                                JProgressBar progressBar) {
        int row = downloadRowIndex++;
        JButton checkButton = createIconButton("✓", "Check / validate artifact status (never writes)");
        JButton convertButton = new JButton("Check first");
        convertButton.setEnabled(false);
        convertButton.setToolTipText("Press Check to inspect, then Convert if a package is needed");
        JLabel statusLabel = new JLabel(artifactRow.family().displayName() + " — not checked");
        statusLabel.setFont(statusLabel.getFont().deriveFont(Font.PLAIN, 11f));

        RowControls controls = new RowControls(artifactRow, convertButton, statusLabel);
        rowControls.add(controls);
        checkButton.addActionListener(e -> refreshRow(controls));
        convertButton.addActionListener(e -> startConvert(controls));

        rows.add(downloadButton, rowConstraints(row, 0, 0.55d, GridBagConstraints.HORIZONTAL));
        rows.add(configButton, rowConstraints(row, 1, 0.0d, GridBagConstraints.NONE));
        rows.add(checkButton, rowConstraints(row, 2, 0.0d, GridBagConstraints.NONE));
        rows.add(convertButton, rowConstraints(row, 3, 0.0d, GridBagConstraints.NONE));
        rows.add(openFolderButton, rowConstraints(row, 4, 0.0d, GridBagConstraints.NONE));
        rows.add(statusLabel, rowConstraints(row, 5, 0.45d, GridBagConstraints.HORIZONTAL));
        rows.add(progressBar, rowConstraints(row, 6, 0.6d, GridBagConstraints.HORIZONTAL));
    }

    /** Re-inspect every row's artifact status (manual; never writes/compiles). */
    private void refreshAllRows() {
        for (RowControls controls : rowControls) {
            refreshRow(controls);
        }
    }

    /** Inspect one row off the EDT and update its status label + Convert button. Never writes. */
    private void refreshRow(RowControls controls) {
        controls.statusLabel.setText(controls.row.family().displayName() + " — checking…");
        new SwingWorker<ModelArtifactRow.RowView, Void>() {
            @Override
            protected ModelArtifactRow.RowView doInBackground() {
                return controls.row.refresh();
            }

            @Override
            protected void done() {
                try {
                    applyRowView(controls, get());
                } catch (Exception ex) {
                    controls.statusLabel.setText(controls.row.family().displayName()
                            + " — check failed: " + describe(ex));
                    appendLog("Status check failed for " + controls.row.family().displayName()
                            + ": " + describe(ex));
                }
            }
        }.execute();
    }

    private void applyRowView(RowControls controls, ModelArtifactRow.RowView view) {
        controls.statusLabel.setText(view.statusText());
        controls.convertButton.setText(view.convertLabel());
        controls.convertButton.setEnabled(view.convertEnabled());
        controls.convertButton.setToolTipText(view.convertTooltip());
    }

    /** Convert one row (the only UI write path) off the EDT, then re-inspect. */
    private void startConvert(RowControls controls) {
        appendLog("Convert " + controls.row.family().displayName() + ": " + controls.row.modelDir());
        controls.convertButton.setEnabled(false);
        new SwingWorker<ModelConversionResult, Void>() {
            @Override
            protected ModelConversionResult doInBackground() {
                return controls.row.convert();
            }

            @Override
            protected void done() {
                try {
                    ModelConversionResult result = get();
                    appendLog((result.ok() ? "  OK: " : "  FAILED: ") + result.message()
                            + (result.output() != null ? " -> " + result.output() : ""));
                } catch (Exception ex) {
                    appendLog("  Convert error: " + describe(ex));
                } finally {
                    refreshRow(controls);
                }
            }
        }.execute();
    }

    /** Per-row UI handles bound to a {@link ModelArtifactRow}. */
    private record RowControls(ModelArtifactRow row, JButton convertButton, JLabel statusLabel) {
    }

    private GridBagConstraints rowConstraints(int row, int column, double weightX, int fill) {
        GridBagConstraints constraints = new GridBagConstraints();
        constraints.gridx = column;
        constraints.gridy = row;
        constraints.weightx = weightX;
        constraints.fill = fill;
        constraints.anchor = GridBagConstraints.CENTER;
        constraints.insets = new Insets(2, 3, 2, 3);
        return constraints;
    }

    private JProgressBar registerProgressBar(ModelDownloadManifest manifest) {
        JProgressBar progressBar = createDownloadProgressBar();
        downloadProgressBars.put(manifest.modelId(), progressBar);
        return progressBar;
    }

    private JProgressBar createDownloadProgressBar() {
        JProgressBar progressBar = new JProgressBar(0, 100);
        progressBar.setStringPainted(true);
        progressBar.setString("Idle");
        progressBar.setValue(0);
        progressBar.setPreferredSize(new Dimension(300, 24));
        return progressBar;
    }

    private JButton createConfigButton(String modelId) {
        JButton button = createIconButton(SETTINGS_BUTTON_ICON, "Configure download URLs");
        button.getAccessibleContext().setAccessibleName("Configure download URLs");
        button.addActionListener(e -> openConfigDialog(modelId));
        return button;
    }

    private JButton createQwenConfigButton() {
        JButton button = createIconButton(SETTINGS_BUTTON_ICON, "Configure Qwen download source, variant and URLs");
        button.getAccessibleContext().setAccessibleName("Configure Qwen download settings");
        button.addActionListener(e -> openQwenConfigDialog());
        return button;
    }

    private JButton createAccessConfigButton(String modelId) {
        JButton button = createIconButton(SETTINGS_BUTTON_ICON, "Configure gated download URLs and HF token");
        button.getAccessibleContext().setAccessibleName("Configure gated download settings");
        button.addActionListener(e -> openAccessConfigDialog(modelId));
        return button;
    }

    private JButton createOpenFolderButton(Supplier<Path> folderSupplier) {
        JButton button = createIconButton("\uD83D\uDCC2", "Open target folder");
        button.getAccessibleContext().setAccessibleName("Open target folder");
        button.addActionListener(e -> {
            Path folder = folderSupplier.get();
            DownloadFolderOpener.openFolder(folder, this::appendLog);
        });
        return button;
    }

    private JButton createIconButton(String text, String tooltip) {
        JButton button = new JButton(text);
        button.setToolTipText(tooltip);
        button.setMargin(ICON_BUTTON_MARGIN);
        button.setIconTextGap(0);
        Dimension square = new Dimension(ICON_BUTTON_SIZE, ICON_BUTTON_SIZE);
        button.setPreferredSize(square);
        button.setMinimumSize(square);
        button.setMaximumSize(square);
        button.setFocusable(false);
        return button;
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

    private void openQwenConfigDialog() {
        QwenDownloadSettingsDialog dialog = new QwenDownloadSettingsDialog(SwingUtilities.getWindowAncestor(this));
        dialog.setVisible(true);
    }

    private void openAccessConfigDialog(String modelId) {
        ModelDownloadManifest manifest = manifests.get(modelId);
        if (manifest == null) {
            return;
        }

        DownloadAccessConfigDialog dialog = new DownloadAccessConfigDialog(
                SwingUtilities.getWindowAncestor(this), manifest, overrideStore.accessSettings(modelId));
        dialog.setVisible(true);

        if (dialog.isAccepted()) {
            ModelDownloadManifest updated = manifest.withAllUrls(dialog.getEditedUrls());
            DownloadAccessSettings accessSettings = dialog.getAccessSettings();
            manifests.put(modelId, updated);
            overrideStore.storeOverrides(updated);
            overrideStore.storeAccessSettings(modelId, accessSettings);
            appendLog("Updated gated download settings for: " + modelId
                    + (accessSettings.hasHuggingFaceToken() ? " (HF token configured)" : " (HF token cleared)"));
        }
    }

    private void startManifestDownload(String modelId) {
        ModelDownloadManifest manifest = manifests.get(modelId);
        if (manifest == null) {
            return;
        }
        startDownload(manifest, modelId);
    }

    /**
     * Download the Qwen 0.5B runtime source: the selected q4f16/ONNX variant into the directml-int4
     * directory - the same directory the Convert flow and the runtime ({@code ModelRuntimeRegistry
     * .qwen05b()}) use. The BF16 SafeTensors path is never the default executable Qwen-0.5B download.
     */
    private void startQwenRuntimeDownload() {
        QwenOnnxModelVariant variant = selectedQwenVariant(); // defaults to q4f16
        QwenModelDownloadConfig config = QwenModelDownloadConfig.forVariant(variant);
        ModelDownloadManifest manifest = overrideStore.applyOverrides(ModelDownloadUrls.manifestForQwen(config));
        manifests.put(manifest.modelId(), manifest);
        if (qwenDownloadProgressBar != null) {
            downloadProgressBars.put(manifest.modelId(), qwenDownloadProgressBar);
        }
        appendLog("Selected Qwen runtime source (q4f16/ONNX): " + ModelDownloadUrls.selectedQwenModelUrl(config));
        appendLog("Runtime package target after Convert: " + runtimeRegistry.qwen05b().runtimePackagePath());
        startDownload(manifest, manifest.modelId() + " (" + variant.modelFileName() + ")");
    }

    private ModelDownloadManifest createQwenManifest(QwenDownloadSource source,
                                                     QwenOnnxModelVariant variant,
                                                     boolean downloadAllVariants) {
        if (source == QwenDownloadSource.SAFETENSORS) {
            return ModelDownloadUrls.manifestForQwenSafeTensors();
        }
        if (downloadAllVariants) {
            return ModelDownloadUrls.manifestForAllQwenVariants();
        }
        return ModelDownloadUrls.manifestForQwen(QwenModelDownloadConfig.forVariant(variant));
    }

    private QwenDownloadSource selectedQwenSource() {
        return model.getQwenDownloadSource();
    }

    private QwenOnnxModelVariant selectedQwenVariant() {
        return QwenOnnxModelVariant.fromModelFileName(model.getQwenModelFile());
    }

    private Path selectedT5TargetDir(String localDirName) {
        return model.getModelRoot().resolve(localDirName);
    }

    private void startDownload(ModelDownloadManifest manifest, String label) {
        boolean force = forceCheckbox.isSelected();
        Path targetDir = model.getModelRoot().resolve(manifest.localDirName());
        JProgressBar progressBar = progressBarFor(manifest);
        resetProgress(progressBar);
        appendLog("Starting download: " + label + " -> " + targetDir);

        new SwingWorker<Boolean, DownloadUiEvent>() {
            @Override
            protected Boolean doInBackground() {
                DownloadProgressTracker progressTracker = new DownloadProgressTracker(manifest.files().size());
                publish(DownloadUiEvent.progress(0, "0%"));
                try {
                    ModelDownloader.downloadFromManifest(manifest, targetDir, force,
                            message -> publish(DownloadUiEvent.message(message)),
                            event -> publish(progressTracker.update(event)),
                            model.getProxyConfiguration(),
                            accessSettingsFor(manifest));
                    List<String> missing = ModelDownloader.missingRequiredFiles(manifest, targetDir);
                    if (missing.isEmpty()) {
                        publish(DownloadUiEvent.message("  Verified: all required files present in " + targetDir));
                    } else {
                        publish(DownloadUiEvent.message("  WARNING: download incomplete - missing/empty "
                                + "required file(s) " + missing + " in " + targetDir
                                + ". Re-run with 'Force re-download' enabled, or delete the folder and retry."));
                    }
                    publish(DownloadUiEvent.progress(100, "Done"));
                    return true;
                } catch (Exception ex) {
                    publish(DownloadUiEvent.message("ERROR: " + describe(ex)));
                    publish(DownloadUiEvent.progress(progressTracker.percent(), "Error"));
                    return false;
                }
            }

            @Override
            protected void process(java.util.List<DownloadUiEvent> chunks) {
                for (DownloadUiEvent event : chunks) {
                    if (event.message() != null) {
                        appendLog(event.message());
                    }
                    if (event.progressValue() != null) {
                        updateProgress(progressBar, event.progressValue(), event.progressText());
                    }
                }
            }

            @Override
            protected void done() {
                try {
                    boolean ok = get();
                    updateProgress(progressBar, ok ? 100 : progressBar.getValue(), ok ? "Done" : "Error");
                    appendLog((ok ? "Download finished: " : "Download ended with errors: ") + label);
                } catch (Exception ex) {
                    updateProgress(progressBar, progressBar.getValue(), "Error");
                    appendLog("Download ended with errors: " + label + " (" + describe(ex) + ")");
                }
            }
        }.execute();
    }

    private DownloadAccessSettings accessSettingsFor(ModelDownloadManifest manifest) {
        DownloadAccessSettings storedSettings = overrideStore.accessSettings(manifest.modelId());
        if (storedSettings.hasHuggingFaceToken()) {
            return storedSettings;
        }
        return new DownloadAccessSettings(System.getenv("HF_TOKEN"));
    }

    private JProgressBar progressBarFor(ModelDownloadManifest manifest) {
        JProgressBar progressBar = downloadProgressBars.get(manifest.modelId());
        if (progressBar != null) {
            return progressBar;
        }
        if (isQwenManifest(manifest)) {
            return qwenDownloadProgressBar;
        }
        return null;
    }

    private boolean isQwenManifest(ModelDownloadManifest manifest) {
        String modelId = manifest.modelId();
        return QwenModelDownloadConfig.LOCAL_DIR_NAME.equals(modelId)
                || ModelDownloadUrls.QWEN_SAFETENSORS_LOCAL_DIR.equals(modelId);
    }

    private void resetProgress(JProgressBar progressBar) {
        if (progressBar == null) {
            return;
        }
        updateProgress(progressBar, 0, "0%");
    }

    private void updateProgress(JProgressBar progressBar, int value, String text) {
        if (progressBar == null) {
            return;
        }
        SwingUtilities.invokeLater(() -> {
            int clamped = Math.max(0, Math.min(100, value));
            progressBar.setIndeterminate(false);
            progressBar.setValue(clamped);
            progressBar.setString(text);
        });
    }

    private void copyUrlToClipboard(String url, String label) {
        StringSelection selection = new StringSelection(url == null ? "" : url);
        try {
            Toolkit.getDefaultToolkit().getSystemClipboard().setContents(selection, null);
            appendLog("Copied URL" + label + ": " + url);
        } catch (HeadlessException | IllegalStateException | SecurityException ex) {
            JOptionPane.showMessageDialog(this,
                    "Could not copy URL to clipboard: " + ex.getMessage(),
                    "Clipboard unavailable",
                    JOptionPane.WARNING_MESSAGE);
        }
    }

    private record DownloadUiEvent(String message, Integer progressValue, String progressText) {
        static DownloadUiEvent message(String message) {
            return new DownloadUiEvent(message, null, null);
        }

        static DownloadUiEvent progress(int value, String text) {
            return new DownloadUiEvent(null, value, text);
        }
    }

    private static final class DownloadProgressTracker {
        private final int totalFiles;
        private int lastPercent;

        DownloadProgressTracker(int totalFiles) {
            this.totalFiles = Math.max(0, totalFiles);
            this.lastPercent = 0;
        }

        DownloadUiEvent update(ModelDownloader.ProgressEvent event) {
            if (event == null) {
                return DownloadUiEvent.progress(lastPercent, lastPercent + "%");
            }
            lastPercent = event.aggregatePercent();
            return DownloadUiEvent.progress(lastPercent, progressText(event, lastPercent));
        }

        int percent() {
            return totalFiles == 0 ? 100 : lastPercent;
        }

        private static String progressText(ModelDownloader.ProgressEvent event, int percent) {
            if (event.skipped()) {
                return percent + "% skipped " + event.localFilename();
            }
            if (event.completed()) {
                return percent + "% done " + event.localFilename();
            }
            if (event.totalBytes() > 0L) {
                int filePercent = (int) Math.round(Math.max(0.0d, Math.min(100.0d,
                        (event.bytesRead() * 100.0d) / event.totalBytes())));
                return percent + "% " + event.localFilename() + " "
                        + filePercent + "% (" + formatBytes(event.bytesRead())
                        + " / " + formatBytes(event.totalBytes()) + ")";
            }
            return percent + "% " + event.localFilename() + " " + formatBytes(event.bytesRead());
        }

        private static String formatBytes(long bytes) {
            if (bytes < 1024L) {
                return bytes + " B";
            }
            double kib = bytes / 1024.0d;
            if (kib < 1024.0d) {
                return String.format(java.util.Locale.ROOT, "%.1f KiB", kib);
            }
            double mib = kib / 1024.0d;
            if (mib < 1024.0d) {
                return String.format(java.util.Locale.ROOT, "%.1f MiB", mib);
            }
            return String.format(java.util.Locale.ROOT, "%.2f GiB", mib / 1024.0d);
        }
    }

    private final class QwenDownloadSettingsDialog extends JDialog {

        private final JComboBox<QwenDownloadSource> sourceSelector;
        private final JComboBox<QwenOnnxModelVariant> variantSelector;
        private final JCheckBox allVariantsCheckbox;
        private final JPanel filesPanel;
        private final List<JTextField> urlFields = new ArrayList<JTextField>();

        private ModelDownloadManifest currentManifest;

        QwenDownloadSettingsDialog(Window owner) {
            super(owner, "Configure Qwen2.5-Coder 0.5B", ModalityType.APPLICATION_MODAL);
            setDefaultCloseOperation(DISPOSE_ON_CLOSE);

            sourceSelector = new JComboBox<QwenDownloadSource>(QwenDownloadSource.values());
            sourceSelector.setSelectedItem(selectedQwenSource());
            variantSelector = new JComboBox<QwenOnnxModelVariant>(QwenOnnxModelVariant.values());
            variantSelector.setSelectedItem(selectedQwenVariant());
            allVariantsCheckbox = new JCheckBox("download all ONNX variants");
            allVariantsCheckbox.setSelected(downloadAllQwenVariants);
            filesPanel = new JPanel(new GridBagLayout());

            buildUserInterface();
            updateManifestRows();
            pack();
            setMinimumSize(new Dimension(820, 360));
            setLocationRelativeTo(owner);
        }

        private void buildUserInterface() {
            JPanel contentPanel = new JPanel(new BorderLayout(8, 8));
            contentPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

            JPanel selectorPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 2));
            selectorPanel.add(new JLabel("Quelle:"));
            selectorPanel.add(sourceSelector);
            selectorPanel.add(new JLabel("ONNX-Variante:"));
            selectorPanel.add(variantSelector);
            selectorPanel.add(allVariantsCheckbox);
            contentPanel.add(selectorPanel, BorderLayout.NORTH);

            sourceSelector.addActionListener(e -> updateManifestRows());
            variantSelector.addActionListener(e -> updateManifestRows());
            allVariantsCheckbox.addActionListener(e -> updateManifestRows());

            JPanel filesWrapper = new JPanel(new BorderLayout(4, 4));
            filesWrapper.setBorder(BorderFactory.createTitledBorder("Effective download URLs"));
            filesWrapper.add(new JScrollPane(filesPanel), BorderLayout.CENTER);
            contentPanel.add(filesWrapper, BorderLayout.CENTER);

            JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT));
            JButton okButton = new JButton("OK");
            okButton.addActionListener(e -> accept());
            JButton cancelButton = new JButton("Cancel");
            cancelButton.addActionListener(e -> dispose());
            buttonPanel.add(okButton);
            buttonPanel.add(cancelButton);
            contentPanel.add(buttonPanel, BorderLayout.SOUTH);

            setContentPane(contentPanel);
        }

        private void updateManifestRows() {
            QwenDownloadSource source = selectedDialogSource();
            boolean onnxSelected = source == QwenDownloadSource.ONNX;
            variantSelector.setEnabled(onnxSelected);
            allVariantsCheckbox.setEnabled(onnxSelected);
            if (!onnxSelected && allVariantsCheckbox.isSelected()) {
                allVariantsCheckbox.setSelected(false);
            }

            currentManifest = overrideStore.applyOverrides(
                    createQwenManifest(source, selectedDialogVariant(), onnxSelected && allVariantsCheckbox.isSelected()));
            rebuildFileRows();
        }

        private void rebuildFileRows() {
            filesPanel.removeAll();
            urlFields.clear();

            GridBagConstraints constraints = new GridBagConstraints();
            constraints.insets = new Insets(2, 2, 2, 2);
            constraints.fill = GridBagConstraints.HORIZONTAL;

            int row = 0;
            for (ModelFileDescriptor descriptor : currentManifest.files()) {
                String labelText = descriptor.displayName()
                        + (descriptor.required() ? " (required)" : " (optional)");
                constraints.gridx = 0;
                constraints.gridy = row;
                constraints.weightx = 0.0d;
                constraints.fill = GridBagConstraints.NONE;
                constraints.anchor = GridBagConstraints.WEST;
                filesPanel.add(new JLabel(labelText), constraints);

                JTextField urlField = new JTextField(descriptor.currentUrl(), 52);
                urlFields.add(urlField);
                constraints.gridx = 1;
                constraints.weightx = 1.0d;
                constraints.fill = GridBagConstraints.HORIZONTAL;
                filesPanel.add(urlField, constraints);

                final int fieldIndex = row;
                JButton copyButton = createIconButton("\uD83D\uDCCB", "Copy address");
                copyButton.getAccessibleContext().setAccessibleName("Copy URL for " + descriptor.localFilename());
                copyButton.addActionListener(e -> copyUrlToClipboard(urlFields.get(fieldIndex).getText(),
                        " for " + descriptor.localFilename()));
                constraints.gridx = 2;
                constraints.weightx = 0.0d;
                constraints.fill = GridBagConstraints.NONE;
                filesPanel.add(copyButton, constraints);

                JButton browserButton = createIconButton("\uD83C\uDF10", "Open address in default browser");
                browserButton.getAccessibleContext().setAccessibleName("Open URL for " + descriptor.localFilename());
                browserButton.addActionListener(e -> DownloadUrlOpener.openInBrowser(
                        urlFields.get(fieldIndex).getText(), this));
                constraints.gridx = 3;
                filesPanel.add(browserButton, constraints);

                row++;
            }

            filesPanel.revalidate();
            filesPanel.repaint();
        }

        private void accept() {
            ModelDownloadManifest updated = currentManifest.withAllUrls(readEditedUrls());
            QwenDownloadSource source = selectedDialogSource();
            model.setQwenDownloadSource(source);
            model.setQwenModelFile(selectedDialogVariant().modelFileName());
            downloadAllQwenVariants = source == QwenDownloadSource.ONNX && allVariantsCheckbox.isSelected();

            manifests.put(updated.modelId(), updated);
            overrideStore.storeOverrides(updated);
            appendLog("Updated Qwen download settings for: " + updated.modelId());
            dispose();
        }

        private List<String> readEditedUrls() {
            ArrayList<String> urls = new ArrayList<String>();
            for (JTextField field : urlFields) {
                urls.add(field.getText().trim());
            }
            return urls;
        }

        private QwenDownloadSource selectedDialogSource() {
            Object selected = sourceSelector.getSelectedItem();
            if (selected instanceof QwenDownloadSource) {
                return (QwenDownloadSource) selected;
            }
            return QwenDownloadSource.ONNX;
        }

        private QwenOnnxModelVariant selectedDialogVariant() {
            Object selected = variantSelector.getSelectedItem();
            if (selected instanceof QwenOnnxModelVariant) {
                return (QwenOnnxModelVariant) selected;
            }
            return QwenOnnxModelVariant.Q4F16;
        }
    }

    private static String describe(Throwable throwable) {
        if (throwable == null) {
            return "unknown error";
        }
        String message = throwable.getMessage();
        if (message == null || message.trim().isEmpty()) {
            return throwable.getClass().getName();
        }
        return throwable.getClass().getName() + ": " + message;
    }

    private void appendLog(String message) {
        SwingUtilities.invokeLater(() -> {
            logArea.append(message + "\n");
            logArea.setCaretPosition(logArea.getDocument().getLength());
        });
    }
}
