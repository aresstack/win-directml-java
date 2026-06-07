package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.inference.qwen.QwenWdmlPackCompileTool;
import com.aresstack.windirectml.inference.t5.T5CompileOptions;
import com.aresstack.windirectml.inference.t5.T5InferenceEngine;
import com.aresstack.windirectml.inference.t5.T5WdmlPackCompiler;
import com.aresstack.windirectml.workbench.WorkbenchModel;
import com.aresstack.windirectml.workbench.download.DownloadFolderOpener;
import com.aresstack.windirectml.workbench.download.DownloadOverrideStore;
import com.aresstack.windirectml.workbench.download.ModelDownloadManifest;
import com.aresstack.windirectml.workbench.download.ModelDownloadUrls;
import com.aresstack.windirectml.workbench.download.ModelDownloader;
import com.aresstack.windirectml.workbench.download.QwenDownloadSource;
import com.aresstack.windirectml.workbench.download.QwenModelDownloadConfig;
import com.aresstack.windirectml.workbench.download.QwenOnnxModelVariant;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
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
    private final Map<String, JProgressBar> downloadProgressBars = new HashMap<String, JProgressBar>();

    private JComboBox<QwenDownloadSource> qwenSourceSelector;
    private JComboBox<QwenOnnxModelVariant> qwenVariantSelector;
    private JTextField qwenSelectedUrlField;
    private JCheckBox downloadAllQwenVariantsCheckbox;
    private JProgressBar qwenDownloadProgressBar;

    public DownloadPanel(WorkbenchModel model) {
        this.model = model;
        this.overrideStore = new DownloadOverrideStore(DownloadOverrideStore.defaultStorePath());
        setLayout(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        JPanel buttons = new JPanel(new GridLayout(0, 4, 4, 4));

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
        addManifestRow(buttons, "Download Qwen2.5-Coder 1.5B Instruct (SafeTensors, planned)",
                ModelDownloadUrls.manifestForQwenCoder1_5BSafeTensors());
        addManifestRow(buttons, "Download Qwen2.5-Coder 3B Instruct (SafeTensors, planned)",
                ModelDownloadUrls.manifestForQwenCoder3BSafeTensors());
        addManifestRow(buttons, "Download SmolLM2 135M Instruct (Summarizer planned)",
                ModelDownloadUrls.manifestForSmolLm2_135M());
        addManifestRow(buttons, "Download SmolLM2 360M Instruct (Summarizer planned)",
                ModelDownloadUrls.manifestForSmolLm2_360M());
        addT5Row(buttons, "Download google/flan-t5-small (SafeTensors)",
                ModelDownloadUrls.manifestForGoogleFlanT5Small(),
                "Compile FLAN-T5 small SafeTensors → wdmlpack");
        addT5Row(buttons, "Download google-t5/t5-small (SafeTensors smoke-test)",
                ModelDownloadUrls.manifestForGoogleT5Small(),
                "Compile T5-small SafeTensors → wdmlpack");
        addCodeT5Row(buttons);

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

        buttons.add(registerProgressBar(manifest));
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

        buttons.add(registerProgressBar(manifest));
        buttons.add(createConfigButton(modelId));
        buttons.add(createOpenFolderButton(() -> model.getModelRoot().resolve(manifest.localDirName())));
    }

    private void addManifestRow(JPanel buttons, String label, ModelDownloadManifest manifest) {
        ModelDownloadManifest effectiveManifest = overrideStore.applyOverrides(manifest);
        manifests.put(effectiveManifest.modelId(), effectiveManifest);
        String modelId = effectiveManifest.modelId();

        JButton downloadBtn = new JButton(label);
        downloadBtn.addActionListener(e -> startManifestDownload(modelId));
        buttons.add(downloadBtn);

        buttons.add(registerProgressBar(effectiveManifest));
        buttons.add(createConfigButton(modelId));
        buttons.add(createOpenFolderButton(() -> model.getModelRoot().resolve(effectiveManifest.localDirName())));
    }

    private void addCodeT5Row(JPanel buttons) {
        ModelDownloadManifest smallManifest = ModelDownloadUrls.manifestForCodeT5Small();
        addT5Row(buttons, "Download CodeT5 small checkpoint", smallManifest,
                "Compile CodeT5 small → wdmlpack");

        ModelDownloadManifest multiSumManifest = ModelDownloadUrls.manifestForCodeT5BaseMultiSum();
        addT5Row(buttons, "Download CodeT5 base multi-sum checkpoint", multiSumManifest,
                "Compile CodeT5 base multi-sum → wdmlpack");
    }

    private void addT5Row(JPanel buttons, String downloadLabel, ModelDownloadManifest baseManifest, String compileLabel) {
        ModelDownloadManifest manifest = overrideStore.applyOverrides(baseManifest);
        manifests.put(manifest.modelId(), manifest);
        String modelId = manifest.modelId();

        JButton downloadBtn = new JButton(downloadLabel);
        downloadBtn.setToolTipText("Downloads T5 config/tokenizer files and model weights when the upstream repository provides a supported source.");
        downloadBtn.addActionListener(e -> startManifestDownload(modelId));
        buttons.add(downloadBtn);

        buttons.add(registerProgressBar(manifest));

        JButton compileButton = new JButton(compileLabel);
        compileButton.setToolTipText("Compile model.safetensors or pytorch_model.bin into model_t5.wdmlpack in this model folder.");
        compileButton.addActionListener(e -> startT5ModelPackageCompile(manifest.localDirName(), manifest.modelId()));
        buttons.add(compileButton);

        buttons.add(createOpenFolderButton(() -> selectedT5TargetDir(manifest.localDirName())));
    }

    private JPanel createQwenPanel() {
        JPanel panel = new JPanel(new BorderLayout(4, 4));
        panel.setBorder(BorderFactory.createTitledBorder("Qwen2.5-Coder 0.5B"));

        JPanel controls = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 2));
        controls.add(new JLabel("Quelle:"));

        qwenSourceSelector = new JComboBox<QwenDownloadSource>(QwenDownloadSource.values());
        qwenSourceSelector.setSelectedItem(model.getQwenDownloadSource());
        qwenSourceSelector.addActionListener(e -> updateSelectedQwenSource());
        controls.add(qwenSourceSelector);

        controls.add(new JLabel("ONNX-Variante:"));
        qwenVariantSelector = new JComboBox<QwenOnnxModelVariant>(QwenOnnxModelVariant.values());
        qwenVariantSelector.setSelectedItem(QwenOnnxModelVariant.fromModelFileName(model.getQwenModelFile()));
        qwenVariantSelector.addActionListener(e -> updateSelectedQwenSource());
        controls.add(qwenVariantSelector);

        JButton downloadBtn = new JButton("Download selected Qwen");
        downloadBtn.addActionListener(e -> startQwenDownload());
        controls.add(downloadBtn);

        JButton compileBtn = new JButton("Compile SafeTensors → wdmlpack");
        compileBtn.setToolTipText("Compile downloaded Qwen SafeTensors into the selected runtime package name.");
        compileBtn.addActionListener(e -> startQwenSafeTensorsCompile());
        controls.add(compileBtn);

        downloadAllQwenVariantsCheckbox = new JCheckBox("alle ONNX-Varianten herunterladen");
        downloadAllQwenVariantsCheckbox.setToolTipText("Nur aktivieren, wenn alle Qwen-ONNX-Dateien vorab geladen werden sollen.");
        controls.add(downloadAllQwenVariantsCheckbox);

        controls.add(createOpenFolderButton(this::selectedQwenTargetDir));
        panel.add(controls, BorderLayout.NORTH);

        JPanel qwenProgressPanel = new JPanel(new BorderLayout(4, 4));
        qwenProgressPanel.add(new JLabel("Download-Fortschritt:"), BorderLayout.WEST);
        qwenDownloadProgressBar = createDownloadProgressBar();
        qwenProgressPanel.add(qwenDownloadProgressBar, BorderLayout.CENTER);
        panel.add(qwenProgressPanel, BorderLayout.CENTER);

        JPanel urlPanel = new JPanel(new BorderLayout(4, 4));
        urlPanel.add(new JLabel("Gewählte HF-URL:"), BorderLayout.WEST);
        qwenSelectedUrlField = new JTextField();
        qwenSelectedUrlField.setEditable(false);
        urlPanel.add(qwenSelectedUrlField, BorderLayout.CENTER);
        JButton copyButton = new JButton("Copy");
        copyButton.addActionListener(e -> copySelectedQwenUrl());
        urlPanel.add(copyButton, BorderLayout.EAST);
        panel.add(urlPanel, BorderLayout.SOUTH);

        updateSelectedQwenSource();
        return panel;
    }

    private void updateSelectedQwenSource() {
        QwenDownloadSource source = selectedQwenSource();
        model.setQwenDownloadSource(source);
        QwenOnnxModelVariant variant = selectedQwenVariant();
        model.setQwenModelFile(variant.modelFileName());

        boolean onnx = source == QwenDownloadSource.ONNX;
        if (qwenVariantSelector != null) {
            qwenVariantSelector.setEnabled(onnx);
        }
        if (downloadAllQwenVariantsCheckbox != null) {
            downloadAllQwenVariantsCheckbox.setEnabled(onnx);
            if (!onnx) {
                downloadAllQwenVariantsCheckbox.setSelected(false);
            }
        }
        if (qwenSelectedUrlField != null) {
            if (onnx) {
                QwenModelDownloadConfig config = QwenModelDownloadConfig.forVariant(variant);
                qwenSelectedUrlField.setText(ModelDownloadUrls.selectedQwenModelUrl(config));
            } else {
                qwenSelectedUrlField.setText(ModelDownloadUrls.selectedQwenSafeTensorsModelUrl());
            }
        }
    }

    private QwenDownloadSource selectedQwenSource() {
        if (qwenSourceSelector == null) {
            return model.getQwenDownloadSource();
        }
        Object selected = qwenSourceSelector.getSelectedItem();
        if (selected instanceof QwenDownloadSource source) {
            return source;
        }
        return QwenDownloadSource.ONNX;
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
        QwenDownloadSource source = selectedQwenSource();
        QwenOnnxModelVariant variant = selectedQwenVariant();
        model.setQwenDownloadSource(source);
        model.setQwenModelFile(variant.modelFileName());

        ModelDownloadManifest manifest;
        String label;
        if (source == QwenDownloadSource.SAFETENSORS) {
            manifest = ModelDownloadUrls.manifestForQwenSafeTensors();
            label = manifest.modelId() + " (SafeTensors)";
            appendLog("Selected Qwen SafeTensors URL: " + ModelDownloadUrls.selectedQwenSafeTensorsModelUrl());
            appendLog("Runtime package target after compile: " + selectedQwenRuntimePackagePath());
        } else if (downloadAllQwenVariantsCheckbox != null && downloadAllQwenVariantsCheckbox.isSelected()) {
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

    private void startQwenSafeTensorsCompile() {
        Path targetDir = selectedQwenTargetDir();
        Path output = selectedQwenRuntimePackagePath();
        appendLog("Compiling Qwen SafeTensors: " + targetDir + " -> " + output);

        new SwingWorker<Boolean, String>() {
            @Override
            protected Boolean doInBackground() {
                try {
                    QwenWdmlPackCompileTool.CompileResult result =
                            QwenWdmlPackCompileTool.compileSafeTensorsDirectory(
                                    new QwenWdmlPackCompileTool.CompileOptions(
                                            targetDir, output, true, false, false, true));
                    publish("  runtimeLoadable=" + result.runtimeLoadable()
                            + ", mode=" + result.runtimeLoadMode()
                            + ", tensors=" + result.tensorCount());
                    publish("  Wrote: " + result.output());
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
                    appendLog(get() ? "Qwen SafeTensors compile finished."
                            : "Qwen SafeTensors compile ended with errors.");
                } catch (Exception ex) {
                    appendLog("Qwen SafeTensors compile ended with errors: " + ex.getMessage());
                }
            }
        }.execute();
    }

    private Path selectedQwenTargetDir() {
        if (selectedQwenSource() == QwenDownloadSource.SAFETENSORS) {
            return model.getModelRoot().resolve(ModelDownloadUrls.QWEN_SAFETENSORS_LOCAL_DIR);
        }
        return model.getModelRoot().resolve(QwenModelDownloadConfig.LOCAL_DIR_NAME);
    }

    private Path selectedQwenRuntimePackagePath() {
        String modelFile = model.getQwenModelFile();
        String base = modelFile.endsWith(".onnx")
                ? modelFile.substring(0, modelFile.length() - ".onnx".length())
                : modelFile;
        return selectedQwenTargetDir().resolve(base + ".wdmlpack");
    }

    private void startT5ModelPackageCompile(String localDirName, String modelId) {
        Path targetDir = selectedT5TargetDir(localDirName);
        Path output = selectedT5RuntimePackagePath(localDirName);
        appendLog("Compiling T5 model package (" + modelId + "): " + targetDir + " -> " + output);

        new SwingWorker<Boolean, String>() {
            @Override
            protected Boolean doInBackground() {
                try {
                    if (!hasSupportedT5TensorSource(targetDir)) {
                        publish("ERROR: No supported T5 tensor source found in " + targetDir);
                        publish("  Expected *.safetensors, pytorch_model.bin restricted state_dict checkpoint, or a precompiled "
                                + T5InferenceEngine.DEFAULT_PACKAGE_NAME + ".");
                        return false;
                    }
                    T5WdmlPackCompiler.T5CompileResult result = T5WdmlPackCompiler.compile(
                            new T5CompileOptions(targetDir, output, false, true));
                    publish("  wrote=" + result.written()
                            + ", output=" + result.output()
                            + ", weightsLoadable=" + result.runtimePackage().weightsLoadable()
                            + ", runtimeLoadable=" + result.runtimePackage().runtimeLoadable()
                            + ", mode=" + result.runtimePackage().manifest().get("runtimeLoadMode"));
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
                    appendLog(get() ? "T5 model package compile finished."
                            : "T5 model package compile ended with errors.");
                } catch (Exception ex) {
                    appendLog("T5 model package compile ended with errors: " + ex.getMessage());
                }
            }
        }.execute();
    }

    private static boolean hasSupportedT5TensorSource(Path directory) throws java.io.IOException {
        if (!java.nio.file.Files.isDirectory(directory)) {
            return false;
        }
        if (java.nio.file.Files.isRegularFile(directory.resolve("pytorch_model.bin"))) {
            return true;
        }
        try (java.util.stream.Stream<Path> stream = java.nio.file.Files.list(directory)) {
            return stream
                    .filter(java.nio.file.Files::isRegularFile)
                    .anyMatch(path -> path.getFileName().toString().endsWith(".safetensors"));
        }
    }

    private Path selectedT5TargetDir(String localDirName) {
        return model.getModelRoot().resolve(localDirName);
    }

    private Path selectedT5RuntimePackagePath(String localDirName) {
        return selectedT5TargetDir(localDirName).resolve(T5InferenceEngine.DEFAULT_PACKAGE_NAME);
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
                            model.getProxyConfiguration());
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

    private JProgressBar progressBarFor(ModelDownloadManifest manifest) {
        JProgressBar progressBar = downloadProgressBars.get(manifest.modelId());
        if (progressBar != null) {
            return progressBar;
        }
        return qwenDownloadProgressBar;
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
