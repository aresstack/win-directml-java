package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.config.generation.GenerationModelRegistry;
import com.aresstack.windirectml.config.generation.GenerationModelRegistry.Entry;
import com.aresstack.windirectml.inference.InferenceException;
import com.aresstack.windirectml.inference.InferenceRequest;
import com.aresstack.windirectml.inference.InferenceResult;
import com.aresstack.windirectml.inference.Phi3InferenceEngine;
import com.aresstack.windirectml.inference.Phi3Summarizer;
import com.aresstack.windirectml.inference.Summary;
import com.aresstack.windirectml.inference.SummaryRequest;
import com.aresstack.windirectml.inference.qwen.QwenInferenceEngine;
import com.aresstack.windirectml.inference.qwen.QwenModelDirValidator;
import com.aresstack.windirectml.inference.t5.T5InferenceEngine;
import com.aresstack.windirectml.workbench.WorkbenchModel;
import com.aresstack.windirectml.workbench.runtime.SmolLM2WorkbenchRuntimeRunner;

import javax.swing.*;
import java.awt.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Decoder-backed text generation panel for the DirectML Workbench.
 * <p>
 * The Workbench is the manual test surface for decoder runtimes. Phi-3 uses
 * the summarizer adapter, while Qwen can be selected to exercise the local
 * CPU generation path directly.
 */
public final class SummarizerPanel extends JPanel {

    private static final String QWEN05_MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct";
    private static final String SMOLLM2_MODEL_ID_PREFIX = "HuggingFaceTB/SmolLM2-";

    private final WorkbenchModel model;
    private final JTextArea inputArea;
    private final JTextArea resultArea;
    private final JComboBox<String> modelSelector;
    private final JSpinner maxTokensSpinner;

    public SummarizerPanel(WorkbenchModel model) {
        this.model = model;
        setLayout(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

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

        var statusLabel = new JLabel();
        updateStatusLabel(statusLabel);
        modelSelector.addActionListener(e -> updateStatusLabel(statusLabel));
        modelPanel.add(statusLabel);
        controlsPanel.add(modelPanel, BorderLayout.NORTH);

        var inputPanel = new JPanel(new BorderLayout(4, 4));
        inputPanel.add(new JLabel("Text / prompt:"), BorderLayout.NORTH);
        inputArea = new JTextArea(8, 70);
        inputArea.setLineWrap(true);
        inputArea.setWrapStyleWord(true);
        inputArea.setText("Paste a longer text or prompt here. The workbench will generate output using the selected decoder model.");
        inputPanel.add(new JScrollPane(inputArea), BorderLayout.CENTER);

        var runControls = new JPanel(new FlowLayout(FlowLayout.LEFT));
        runControls.add(new JLabel("Max output tokens:"));
        maxTokensSpinner = new JSpinner(new SpinnerNumberModel(256, 32, 2048, 32));
        runControls.add(maxTokensSpinner);
        var runBtn = new JButton("Generate / Summarize");
        runBtn.addActionListener(e -> runSummarizer());
        runControls.add(runBtn);
        inputPanel.add(runControls, BorderLayout.SOUTH);

        controlsPanel.add(inputPanel, BorderLayout.CENTER);
        add(controlsPanel, BorderLayout.NORTH);

        resultArea = new JTextArea(14, 70);
        resultArea.setEditable(false);
        resultArea.setLineWrap(true);
        resultArea.setWrapStyleWord(true);
        resultArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        add(new JScrollPane(resultArea), BorderLayout.CENTER);
    }

    private String[] buildSummarizerModelOptions() {
        List<String> options = new ArrayList<>();
        for (GenerationModelRegistry.Entry entry : GenerationModelRegistry.entries()) {
            options.add(entry.modelId());
        }
        Entry qwen = GenerationModelRegistry.findByModelId(QWEN05_MODEL_ID);
        if (qwen != null && !options.contains(qwen.modelId())) {
            options.add(qwen.modelId());
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
        if (isQwenTestModel(selected)) {
            label.setText(" 🧪 experimental (DirectML)");
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

        boolean qwenTestModel = isQwenTestModel(selectedModel);
        boolean smolLm2Model = isSmolLm2Model(selectedModel);
        Entry entry = GenerationModelRegistry.findByModelId(selectedModel);
        if (entry != null) {
            if (entry.status() == GenerationModelRegistry.Status.UNSUPPORTED) {
                appendResult("ERROR: Model '" + selectedModel + "' is unsupported in this runtime.");
                return;
            }
            if (entry.status() == GenerationModelRegistry.Status.PLANNED && !qwenTestModel && !smolLm2Model) {
                appendResult("ERROR: Model '" + selectedModel + "' is selectable but not executable yet.");
                appendResult("  Status: planned. Runtime support is in progress for family "
                        + entry.architecture().token() + ".");
                return;
            }
        }

        int maxTokens = (Integer) maxTokensSpinner.getValue();
        appendResult("Loading generation model: " + selectedModel
                + " (backend: " + model.getBackend() + ", maxTokens: " + maxTokens + ")...");
        if (qwenTestModel) {
            appendResult("  NOTE: Qwen acceleration depends on WARP/AUTO and the selected package source (see Config/Download tabs).");
        } else if (smolLm2Model) {
            appendResult("  NOTE: SmolLM2 currently uses the Java reference runtime. First use compiles SafeTensors to model.wdmlpack.");
        } else if (isT5Model(selectedModel)) {
            appendResult("  NOTE: T5-family models use the seq2seq runtime package path (.wdmlpack or SafeTensors auto-compile).");
        }

        new SwingWorker<Void, Void>() {
            @Override
            protected Void doInBackground() {
                try {
                    Path modelDir = resolveSummarizerModelDir(selectedModel);
                    if (qwenTestModel) {
                        runQwenGeneration(modelDir, text, maxTokens, selectedModel);
                    } else if (smolLm2Model) {
                        runSmolLm2Generation(modelDir, text, maxTokens);
                    } else if (isT5Model(selectedModel)) {
                        runT5Generation(modelDir, text, maxTokens, selectedModel);
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

    private void runSmolLm2Generation(Path modelDir, String text, int maxTokens) throws Exception {
        long start = System.nanoTime();
        SmolLM2WorkbenchRuntimeRunner runner = new SmolLM2WorkbenchRuntimeRunner(modelDir);
        appendResult("Initializing SmolLM2 reference runtime from " + modelDir + "...");
        SmolLM2WorkbenchRuntimeRunner.Result result = runner.generate(text, maxTokens);
        appendResult("Model loaded and generated in " + elapsedMs(start) + " ms");
        appendResult("Runtime mode: " + result.runtimeMode());
        appendResult("Runtime package: " + result.packagePath().getFileName());
        appendResult("");
        appendResult("OUTPUT:");
        appendResult(result.text());
        appendResult("");
        appendResult("Generation completed");
        appendResult("  Output tokens: " + result.outputTokens());
        appendResult("  Finish reason: " + result.finishReason());
    }


    private void runT5Generation(Path modelDir, String text, int maxTokens, String selectedModel)
            throws InferenceException {
        validateT5ModelFiles(modelDir);
        long start = System.nanoTime();
        String backend = model.getBackend().name().toLowerCase();
        T5InferenceEngine engine = new T5InferenceEngine(modelDir, maxTokens, backend);
        try {
            appendResult("Initializing T5 runtime package from " + modelDir + " (backend=" + backend + ")...");
            engine.initialize();
            appendResult("Model loaded in " + elapsedMs(start) + " ms");
            appendResult("Seq2Seq generation running with " + engine.executionMode() + ".");
            appendResult("");
            appendResult("OUTPUT:");
            long genStart = System.nanoTime();
            InferenceRequest request = InferenceRequest.builder()
                    .modelId(selectedModel)
                    .systemPrompt("")
                    .userPrompt(text)
                    .maxTokens(maxTokens)
                    .temperature(0.0f)
                    .build();
            InferenceResult result = engine.generate(request);
            appendResult(result.getText());
            appendResult("");
            appendResult("Generation completed in " + elapsedMs(genStart) + " ms");
            if (result.getUsage() != null) {
                appendResult("  Prompt tokens: " + result.getUsage().promptTokens());
                appendResult("  Output tokens: " + result.getUsage().completionTokens());
            }
            appendResult("  Finish reason: " + result.getFinishReason());
        } finally {
            engine.shutdown();
        }
    }

    private void runQwenGeneration(Path modelDir, String text, int maxTokens, String selectedModel)
            throws InferenceException {
        String qwenModelFile = model.getQwenModelFile();
        validateQwenModelFiles(modelDir, qwenModelFile);
        long start = System.nanoTime();
        String backend = model.getBackend().name().toLowerCase();
        QwenInferenceEngine engine = new QwenInferenceEngine(modelDir, maxTokens, backend, qwenModelFile);
        try {
            appendResult("Initializing Qwen runtime (backend=" + backend + ", onnx=" + qwenModelFile + ")...");
            engine.initialize();
            appendResult("Model loaded in " + elapsedMs(start) + " ms");
            appendResult("Prefill running... first token may take a while for long prompts.");
            appendResult("");
            appendResult("OUTPUT:");
            long genStart = System.nanoTime();
            // Format prompt using ChatML template and stream tokens to UI as they appear.
            String systemPrompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
            String formattedPrompt = com.aresstack.windirectml.inference.qwen.QwenChatTemplate.formatChat(
                    systemPrompt, text);
            com.aresstack.windirectml.inference.qwen.Qwen2Runtime runtime = engine.getRuntime();
            final int[] tokenCount = {0};
            String generated = runtime.generateStreaming(formattedPrompt, maxTokens,
                    (tokenId, full, delta) -> {
                        tokenCount[0]++;
                        appendInline(delta);
                    });
            appendResult("");
            appendResult("Generation completed in " + elapsedMs(genStart) + " ms");
            appendResult("  Output tokens (approx): " + tokenCount[0]);
            appendResult("  Finish reason: " + (tokenCount[0] >= maxTokens ? "max_tokens" : "end_turn"));
        } finally {
            engine.shutdown();
        }
    }

    private void appendInline(String s) {
        if (s == null || s.isEmpty()) return;
        SwingUtilities.invokeLater(() -> {
            resultArea.append(s);
            resultArea.setCaretPosition(resultArea.getDocument().getLength());
        });
    }

    private Path resolveSummarizerModelDir(String modelId) {
        Entry entry = GenerationModelRegistry.findByModelId(modelId);
        if (entry != null && !entry.modelDirHints().isEmpty()) {
            for (String hint : entry.modelDirHints()) {
                Path candidate = model.getModelRoot().resolve("..").resolve(hint).normalize();
                if (Files.isDirectory(candidate)) {
                    return candidate;
                }
            }
            for (String hint : entry.modelDirHints()) {
                Path hintPath = Path.of(hint);
                String dirName = hintPath.getFileName().toString();
                Path candidate = model.getModelRoot().resolve(dirName);
                if (Files.isDirectory(candidate)) {
                    return candidate;
                }
            }
        }
        String dirName = modelId.contains("/") ? modelId.substring(modelId.lastIndexOf('/') + 1) : modelId;
        return model.getModelRoot().resolve(dirName);
    }

    private void validatePhi3ModelFiles(Path modelDir) {
        String missing = Phi3InferenceEngine.describeMissingModelFile(modelDir);
        if (missing != null) {
            throw new IllegalStateException(missing + ". Download the model first from the Download tab.");
        }
    }

    private void validateQwenModelFiles(Path modelDir, String modelFileName) {
        String missing = QwenModelDirValidator.describeMissingRequiredFiles(modelDir, modelFileName);
        if (missing != null) {
            throw new IllegalStateException(missing + ". Download the selected Qwen model first from the Download tab.");
        }
    }

    private void validateT5ModelFiles(Path modelDir) {
        String missing = T5InferenceEngine.describeMissingModelFile(modelDir);
        if (missing != null) {
            throw new IllegalStateException(missing + ". Download or compile the selected T5 model first from the Download/Tools flow.");
        }
    }

    private static boolean isQwenTestModel(String modelId) {
        return QWEN05_MODEL_ID.equals(modelId);
    }

    private static boolean isSmolLm2Model(String modelId) {
        return modelId != null && modelId.startsWith(SMOLLM2_MODEL_ID_PREFIX);
    }

    private static boolean isT5Model(String modelId) {
        Entry entry = GenerationModelRegistry.findByModelId(modelId);
        return entry != null && entry.architecture() == GenerationModelRegistry.Architecture.SEQ2SEQ;
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
