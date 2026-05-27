package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.config.generation.GenerationModelRegistry;
import com.aresstack.windirectml.config.generation.GenerationModelRegistry.Entry;
import com.aresstack.windirectml.inference.InferenceException;
import com.aresstack.windirectml.inference.InferenceRequest;
import com.aresstack.windirectml.inference.InferenceResult;
import com.aresstack.windirectml.inference.Phi3InferenceEngine;
import com.aresstack.windirectml.inference.Phi3Summarizer;
import com.aresstack.windirectml.inference.QwenInferenceEngine;
import com.aresstack.windirectml.inference.Summary;
import com.aresstack.windirectml.inference.SummaryRequest;
import com.aresstack.windirectml.inference.qwen.QwenModelDirValidator;
import com.aresstack.windirectml.workbench.WorkbenchModel;

import javax.swing.*;
import java.awt.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Decoder-backed text generation panel for the DirectML Workbench.
 * <p>
 * The default path uses the Phi-3 summarizer adapter. Experimental decoder
 * runtimes, such as Qwen2.5-Coder CPU generation, can be exposed for manual
 * testing behind explicit system-property gates without marking them shipped.
 */
public final class SummarizerPanel extends JPanel {

    private static final String QWEN05_MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct";
    private static final String EXPERIMENTAL_QWEN_PROPERTY = "qwen.enable.experimental.runtime";

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
        modelPanel.add(new JLabel("Generation Model:"));
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
        inputPanel.add(new JLabel("Text / prompt:"), BorderLayout.NORTH);
        inputArea = new JTextArea(8, 70);
        inputArea.setLineWrap(true);
        inputArea.setWrapStyleWord(true);
        inputArea.setText("Paste a longer text or prompt here. The workbench will generate output using the selected decoder model (e.g. Phi-3 or experimental Qwen).");
        inputPanel.add(new JScrollPane(inputArea), BorderLayout.CENTER);

        var runControls = new JPanel(new FlowLayout(FlowLayout.LEFT));
        runControls.add(new JLabel("Max tokens:"));
        maxTokensSpinner = new JSpinner(new SpinnerNumberModel(256, 32, 2048, 32));
        runControls.add(maxTokensSpinner);
        var runBtn = new JButton("Generate / Summarize");
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
        // Use GenerationModelRegistry for generation-capable runnable models.
        List<String> options = new ArrayList<>();
        List<GenerationModelRegistry.Entry> generationModels = GenerationModelRegistry.runnableEntries();
        for (GenerationModelRegistry.Entry entry : generationModels) {
            options.add(entry.modelId());
        }

        // Manual Workbench testing hook for the #99 Qwen CPU runtime.
        // The model remains PLANNED globally and only appears with explicit opt-in.
        if (isExperimentalQwenEnabled()) {
            Entry qwen = GenerationModelRegistry.findByModelId(QWEN05_MODEL_ID);
            if (qwen != null && !options.contains(qwen.modelId())) {
                options.add(qwen.modelId());
            }
        }

        return options.toArray(new String[0]);
    }

    private void updateStatusLabel(JLabel label) {
        String selected = (String) modelSelector.getSelectedItem();
        if (selected == null) {
            label.setText("");
            return;
        }
        Entry entry = GenerationModelRegistry.findByModelId(selected);
        if (entry == null) {
            label.setText(" [unknown]");
            return;
        }
        if (isExperimentalQwen(selected)) {
            label.setText(" 🧪 experimental CPU-only (planned)");
            return;
        }
        String statusText = switch (entry.status()) {
            case SHIPPED -> " ✅ shipped";
            case EXPERIMENTAL -> " 🧪 experimental";
            case PLANNED -> " 🚧 planned";
            case UNSUPPORTED -> " ❌ unsupported";
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
            appendResult("ERROR: No generation model selected.");
            return;
        }

        boolean experimentalQwen = isExperimentalQwen(selectedModel);

        // Check model status
        Entry entry = GenerationModelRegistry.findByModelId(selectedModel);
        if (entry != null) {
            if (entry.status() == GenerationModelRegistry.Status.UNSUPPORTED) {
                appendResult("ERROR: Model '" + selectedModel + "' is unsupported in this runtime.");
                appendResult("  Status: unsupported. This model cannot run locally.");
                return;
            }
            if (entry.status() == GenerationModelRegistry.Status.PLANNED && !experimentalQwen) {
                appendResult("ERROR: Model '" + selectedModel + "' is not yet implemented.");
                appendResult("  Status: planned. Runtime support is in progress.");
                return;
            }
        }

        int maxTokens = (Integer) maxTokensSpinner.getValue();
        appendResult("Loading generation model: " + selectedModel
                + " (backend: " + model.getBackend() + ", maxTokens: " + maxTokens + ")...");

        if (experimentalQwen) {
            appendResult("  NOTE: Qwen CPU runtime is experimental and hidden unless -D"
                    + EXPERIMENTAL_QWEN_PROPERTY + "=true is set.");
        }

        new SwingWorker<Void, Void>() {
            @Override
            protected Void doInBackground() {
                try {
                    Path modelDir = resolveSummarizerModelDir(selectedModel);
                    if (experimentalQwen) {
                        runQwenGeneration(modelDir, text, maxTokens, selectedModel);
                    } else {
                        runPhi3Summarizer(modelDir, text, maxTokens);
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

    private void runPhi3Summarizer(Path modelDir, String text, int maxTokens) throws Exception {
        // Validate required files
        validatePhi3ModelFiles(modelDir);

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
    }

    private void runQwenGeneration(Path modelDir, String text, int maxTokens, String selectedModel)
            throws InferenceException {
        validateQwenModelFiles(modelDir);

        long start = System.nanoTime();
        QwenInferenceEngine engine = new QwenInferenceEngine(modelDir, maxTokens);
        try {
            appendResult("Initializing experimental Qwen CPU runtime...");
            engine.initialize();
            appendResult("Model loaded in " + elapsedMs(start) + " ms");

            long genStart = System.nanoTime();
            InferenceRequest request = InferenceRequest.builder()
                    .modelId(selectedModel)
                    .systemPrompt("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
                    .userPrompt(text)
                    .maxTokens(maxTokens)
                    .build();
            InferenceResult result = engine.generate(request);
            appendResult("Generation completed in " + elapsedMs(genStart) + " ms");
            if (result.getUsage() != null) {
                appendResult("  Prompt tokens: " + result.getUsage().promptTokens());
                appendResult("  Output tokens: " + result.getUsage().completionTokens());
                appendResult("  Total tokens: " + result.getUsage().totalTokens());
            }
            appendResult("  Finish reason: " + result.getFinishReason());
            appendResult("");
            appendResult("OUTPUT:");
            appendResult(result.getText());
        } finally {
            engine.shutdown();
        }
    }

    private Path resolveSummarizerModelDir(String modelId) {
        // Try known directory layouts
        Entry entry = GenerationModelRegistry.findByModelId(modelId);
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

    private void validatePhi3ModelFiles(Path modelDir) {
        String missing = Phi3InferenceEngine.describeMissingModelFile(modelDir);
        if (missing != null) {
            throw new IllegalStateException(missing
                    + ". Download the model first from the Download tab.");
        }
    }

    private void validateQwenModelFiles(Path modelDir) {
        String missing = QwenModelDirValidator.describeMissingModelFile(modelDir);
        if (missing != null) {
            throw new IllegalStateException(missing
                    + ". Download the Qwen model first from the Download tab with -D"
                    + EXPERIMENTAL_QWEN_PROPERTY + "=true.");
        }
    }

    private static boolean isExperimentalQwenEnabled() {
        return Boolean.getBoolean(EXPERIMENTAL_QWEN_PROPERTY);
    }

    private static boolean isExperimentalQwen(String modelId) {
        return isExperimentalQwenEnabled() && QWEN05_MODEL_ID.equals(modelId);
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
