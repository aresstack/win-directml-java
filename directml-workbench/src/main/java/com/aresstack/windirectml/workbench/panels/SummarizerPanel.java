package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.config.models.EmbeddingModelRegistry;
import com.aresstack.windirectml.config.models.EmbeddingModelRegistry.Entry;
import com.aresstack.windirectml.config.models.EmbeddingModelRegistry.UseCase;
import com.aresstack.windirectml.inference.InferenceException;
import com.aresstack.windirectml.inference.Phi3InferenceEngine;
import com.aresstack.windirectml.inference.Phi3Summarizer;
import com.aresstack.windirectml.inference.Summary;
import com.aresstack.windirectml.inference.SummaryRequest;
import com.aresstack.windirectml.workbench.WorkbenchModel;

import javax.swing.*;
import java.awt.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Decoder-backed summarizer panel for the DirectML Workbench.
 * <p>
 * Uses the Phi-3 (or compatible) decoder model to generate real text summaries
 * via the {@link Phi3Summarizer} engine. This replaces the old embedding-based
 * extractive sentence selection approach.
 */
public final class SummarizerPanel extends JPanel {

    private final WorkbenchModel model;
    private final JTextArea inputArea;
    private final JTextArea resultArea;
    private final JComboBox<String> modelSelector;
    private final JSpinner maxTokensSpinner;

    public SummarizerPanel(WorkbenchModel model) {
        this.model = model;
        setLayout(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        // --- Top: model selector and controls ---
        var controlsPanel = new JPanel(new BorderLayout(4, 4));

        var modelPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        modelPanel.add(new JLabel("Summarizer Model:"));
        modelSelector = new JComboBox<>(buildSummarizerModelOptions());
        modelSelector.setSelectedItem(model.getSummarizerModel());
        modelSelector.addActionListener(e -> {
            String selected = (String) modelSelector.getSelectedItem();
            if (selected != null) {
                model.setSummarizerModel(selected);
            }
        });
        modelPanel.add(modelSelector);

        // Status label
        var statusLabel = new JLabel();
        updateStatusLabel(statusLabel);
        modelSelector.addActionListener(e -> updateStatusLabel(statusLabel));
        modelPanel.add(statusLabel);
        controlsPanel.add(modelPanel, BorderLayout.NORTH);

        // Input area
        var inputPanel = new JPanel(new BorderLayout(4, 4));
        inputPanel.add(new JLabel("Text to summarize:"), BorderLayout.NORTH);
        inputArea = new JTextArea(8, 70);
        inputArea.setLineWrap(true);
        inputArea.setWrapStyleWord(true);
        inputArea.setText("Paste a longer text here. The workbench will generate a summary using the selected decoder model (e.g. Phi-3).");
        inputPanel.add(new JScrollPane(inputArea), BorderLayout.CENTER);

        var runControls = new JPanel(new FlowLayout(FlowLayout.LEFT));
        runControls.add(new JLabel("Max tokens:"));
        maxTokensSpinner = new JSpinner(new SpinnerNumberModel(256, 32, 2048, 32));
        runControls.add(maxTokensSpinner);
        var runBtn = new JButton("Summarize");
        runBtn.addActionListener(e -> runSummarizer());
        runControls.add(runBtn);
        inputPanel.add(runControls, BorderLayout.SOUTH);

        controlsPanel.add(inputPanel, BorderLayout.CENTER);
        add(controlsPanel, BorderLayout.NORTH);

        // --- Bottom: result area ---
        resultArea = new JTextArea(14, 70);
        resultArea.setEditable(false);
        resultArea.setLineWrap(true);
        resultArea.setWrapStyleWord(true);
        resultArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        add(new JScrollPane(resultArea), BorderLayout.CENTER);
    }

    private String[] buildSummarizerModelOptions() {
        List<Entry> summarizers = EmbeddingModelRegistry.entriesByUseCase(UseCase.SUMMARIZER);
        String[] options = new String[summarizers.size()];
        for (int i = 0; i < summarizers.size(); i++) {
            options[i] = summarizers.get(i).modelId();
        }
        return options;
    }

    private void updateStatusLabel(JLabel label) {
        String selected = (String) modelSelector.getSelectedItem();
        if (selected == null) {
            label.setText("");
            return;
        }
        Entry entry = EmbeddingModelRegistry.findByModelId(selected);
        if (entry == null) {
            label.setText(" [unknown]");
            return;
        }
        String statusText = switch (entry.status()) {
            case SHIPPED -> " \u2705 shipped";
            case EXPERIMENTAL -> " \uD83E\uDDEA experimental";
            case PLANNED -> " \uD83D\uDEA7 planned";
            case UNSUPPORTED -> " \u274C unsupported";
        };
        label.setText(statusText);
    }

    private void runSummarizer() {
        String text = inputArea.getText().trim();
        if (text.isEmpty()) {
            appendResult("ERROR: Input text is empty.");
            return;
        }

        String selectedModel = (String) modelSelector.getSelectedItem();
        if (selectedModel == null) {
            appendResult("ERROR: No summarizer model selected.");
            return;
        }

        // Check model status
        Entry entry = EmbeddingModelRegistry.findByModelId(selectedModel);
        if (entry != null) {
            if (entry.status() == EmbeddingModelRegistry.Status.UNSUPPORTED) {
                appendResult("ERROR: Model '" + selectedModel + "' is unsupported in this runtime.");
                appendResult("  Status: unsupported. This model cannot run locally.");
                return;
            }
            if (entry.status() == EmbeddingModelRegistry.Status.PLANNED) {
                appendResult("ERROR: Model '" + selectedModel + "' is not yet implemented.");
                appendResult("  Status: planned. Runtime support is in progress.");
                return;
            }
        }

        int maxTokens = (Integer) maxTokensSpinner.getValue();
        appendResult("Loading summarizer model: " + selectedModel
                + " (backend: " + model.getBackend() + ", maxTokens: " + maxTokens + ")...");

        new SwingWorker<Void, Void>() {
            @Override
            protected Void doInBackground() {
                try {
                    Path modelDir = resolveSummarizerModelDir(selectedModel);

                    // Validate required files
                    validateModelFiles(modelDir);

                    String backend = model.getBackend().name().toLowerCase();
                    long start = System.nanoTime();

                    try (var summarizer = new Phi3Summarizer(modelDir, maxTokens, backend)) {
                        appendResult("Initializing model...");
                        summarizer.initialize();
                        appendResult("Model loaded in " + elapsedMs(start) + " ms");

                        long genStart = System.nanoTime();
                        SummaryRequest request = SummaryRequest.of(text, maxTokens);
                        Summary summary = summarizer.summarize(request);
                        appendResult("Generation completed in " + elapsedMs(genStart) + " ms");
                        appendResult("  Prompt tokens: " + summary.promptTokens());
                        appendResult("  Output tokens: " + summary.outputTokens());
                        appendResult("  Finish reason: " + summary.finishReason());
                        appendResult("");
                        appendResult("SUMMARY:");
                        appendResult(summary.text());
                    }
                } catch (InferenceException ex) {
                    appendResult("INFERENCE ERROR: " + ex.getMessage());
                } catch (Exception ex) {
                    appendResult("ERROR: " + ex.getClass().getSimpleName() + ": " + ex.getMessage());
                }
                return null;
            }
        }.execute();
    }

    private Path resolveSummarizerModelDir(String modelId) {
        // Try known directory layouts
        Entry entry = EmbeddingModelRegistry.findByModelId(modelId);
        if (entry != null && !entry.modelDirHints().isEmpty()) {
            for (String hint : entry.modelDirHints()) {
                Path candidate = model.getModelRoot().resolve("..").resolve(hint).normalize();
                if (Files.isDirectory(candidate)) {
                    return candidate;
                }
            }
            // Also try relative to model root directly
            for (String hint : entry.modelDirHints()) {
                Path hintPath = Path.of(hint);
                String dirName = hintPath.getFileName().toString();
                Path candidate = model.getModelRoot().resolve(dirName);
                if (Files.isDirectory(candidate)) {
                    return candidate;
                }
            }
        }

        // Fallback: use the model ID last segment as directory name
        String dirName = modelId.contains("/") ? modelId.substring(modelId.lastIndexOf('/') + 1) : modelId;
        return model.getModelRoot().resolve(dirName);
    }

    private void validateModelFiles(Path modelDir) {
        String missing = Phi3InferenceEngine.describeMissingModelFile(modelDir);
        if (missing != null) {
            throw new IllegalStateException(missing
                    + ". Download the model first from the Download tab.");
        }
    }

    private static long elapsedMs(long startNanos) {
        return (System.nanoTime() - startNanos) / 1_000_000;
    }

    private void appendResult(String message) {
        SwingUtilities.invokeLater(() -> {
            resultArea.append(message + "\n");
            resultArea.setCaretPosition(resultArea.getDocument().getLength());
        });
    }
}
