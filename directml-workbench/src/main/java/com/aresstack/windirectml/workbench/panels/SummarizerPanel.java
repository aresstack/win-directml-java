package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.config.generation.GenerationModelRegistry;
import com.aresstack.windirectml.config.generation.GenerationModelRegistry.Entry;
import com.aresstack.windirectml.inference.GenerationTokenSink;
import com.aresstack.windirectml.inference.GeneratedToken;
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
import com.aresstack.windirectml.inference.smollm2.SmolLM2GenerationProfile;
import com.aresstack.windirectml.inference.prompt.PromptInput;
import com.aresstack.windirectml.inference.prompt.PromptStrategies;
import com.aresstack.windirectml.inference.prompt.PromptTask;
import com.aresstack.windirectml.workbench.WorkbenchModel;
import com.aresstack.windirectml.workbench.runtime.SmolLM2WorkbenchRuntimeRunner;
import com.aresstack.windirectml.workbench.prompt.PromptTaskLabels;

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

    /** Diagnostic toggle ({@code -Dsmollm2.debug.prompt=true}): show the rendered prompt and
     *  effective model config in the output panel. Off by default to keep normal output lean. */
    private static final boolean DEBUG_PROMPT = Boolean.getBoolean("smollm2.debug.prompt");

    private final WorkbenchModel model;
    private final JTextArea inputArea;
    private final JTextArea resultArea;
    private final JComboBox<String> modelSelector;
    private final JComboBox<PromptTask> promptTemplateSelector;
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
                updatePromptTemplateOptions(selected);
            }
        });
        modelPanel.add(modelSelector);

        modelPanel.add(new JLabel("Template:"));
        promptTemplateSelector = new JComboBox<>();
        promptTemplateSelector.setToolTipText("Choose an optional task template. 'Keines' passes the input without an additional task instruction.");
        promptTemplateSelector.setRenderer(new DefaultListCellRenderer() {
            @Override
            public Component getListCellRendererComponent(JList<?> list, Object value, int index,
                                                          boolean isSelected, boolean cellHasFocus) {
                super.getListCellRendererComponent(list, value, index, isSelected, cellHasFocus);
                if (value instanceof PromptTask task) {
                    setText(PromptTaskLabels.labelFor(task));
                }
                return this;
            }
        });
        modelPanel.add(promptTemplateSelector);
        updatePromptTemplateOptions((String) modelSelector.getSelectedItem());
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

    private void updatePromptTemplateOptions(String selectedModel) {
        PromptTask previous = (PromptTask) promptTemplateSelector.getSelectedItem();
        promptTemplateSelector.removeAllItems();
        for (PromptTask task : PromptStrategies.supportedTasks(selectedModel)) {
            promptTemplateSelector.addItem(task);
        }
        PromptTask next = previous == null ? PromptTask.NONE : previous;
        if (!selectPromptTemplate(next)) {
            selectPromptTemplate(PromptTask.NONE);
        }
    }

    private boolean selectPromptTemplate(PromptTask task) {
        for (int i = 0; i < promptTemplateSelector.getItemCount(); i++) {
            if (promptTemplateSelector.getItemAt(i) == task) {
                promptTemplateSelector.setSelectedIndex(i);
                return true;
            }
        }
        return false;
    }

    private void runSummarizer() {
        String text = inputArea.getText().trim();
        if (text.isEmpty()) {
            appendResult("ERROR: Input text is empty.");
            return;
        }

        PromptTask selectedTask = (PromptTask) promptTemplateSelector.getSelectedItem();
        final PromptTask promptTask = selectedTask == null ? PromptTask.NONE : selectedTask;

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
            appendResult("  NOTE: WARP runs SmolLM2's dense projections on the D3D12 software rasterizer (CPU); "
                    + "AUTO uses a hardware GPU when one exists and otherwise falls back to the Java reference "
                    + "runtime. Either way norms/RoPE/attention/KV-cache stay on CPU.");
            appendResult("  NOTE: First use compiles SafeTensors to model.wdmlpack.");
        } else if (isT5Model(selectedModel)) {
            appendResult("  NOTE: T5-family models use the seq2seq runtime package path (.wdmlpack or SafeTensors auto-compile).");
        }

        new SwingWorker<Void, Void>() {
            @Override
            protected Void doInBackground() {
                try {
                    Path modelDir = resolveSummarizerModelDir(selectedModel);
                    if (qwenTestModel) {
                        runQwenGeneration(modelDir, promptTask, text, maxTokens, selectedModel);
                    } else if (smolLm2Model) {
                        runSmolLm2Generation(modelDir, promptTask, text, maxTokens);
                    } else if (isT5Model(selectedModel)) {
                        runT5Generation(modelDir, promptTask, text, maxTokens, selectedModel);
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

    private void runSmolLm2Generation(Path modelDir, PromptTask task, String text, int maxTokens) throws Exception {
        long start = System.nanoTime();
        SmolLM2WorkbenchRuntimeRunner runner = new SmolLM2WorkbenchRuntimeRunner(modelDir);
        appendResult("Initializing SmolLM2 runtime from " + modelDir
                + " (requested backend=" + model.getBackend().name().toLowerCase() + ")...");
        appendResult("");
        if (DEBUG_PROMPT) {
            String renderedPrompt = PromptStrategies.forModel("smollm2").renderPrompt(PromptInput.of(task, text));
            appendResult("[debug] PROMPT (task=" + task + ", " + renderedPrompt.length() + " chars):");
            appendResult(renderedPrompt);
            appendResult("");
        }
        appendResult("OUTPUT:");
        SmolLM2WorkbenchRuntimeRunner.Result result = runner.generate(PromptInput.of(task, text), maxTokens,
                model.getBackend(), new UiTokenSink());
        appendResult("");
        appendResult("Model loaded and generated in " + elapsedMs(start) + " ms");
        appendResult("Requested backend: " + result.requestedBackend());
        appendResult("Runtime mode: " + result.runtimeMode());
        if (!result.fallbackReason().isBlank()) {
            appendResult("Runtime fallback: " + result.fallbackReason());
        }
        for (String warning : result.runtimeWarnings()) {
            appendResult("Runtime warning: " + warning);
        }
        result.warpReadinessReport().ifPresent(this::appendSmolLm2WarpReadiness);
        appendResult("Runtime package: " + result.packagePath().getFileName());
        appendResult("Generation config: greedyChat, repetitionPenalty="
                + com.aresstack.windirectml.inference.smollm2.SmolLM2GenerationOptions.CHAT_REPETITION_PENALTY
                + ", maxTokens=" + maxTokens);
        if (DEBUG_PROMPT) {
            appendResult("  [debug] prompt task: " + task);
            appendResult("  [debug] effective model config: " + result.effectiveConfig());
        }
        if (result.text().isBlank()) {
            appendResult("  NOTE: generated text is empty after detokenization.");
        }
        appendResult("");
        SmolLM2GenerationProfile profile = result.diagnostics().profile();
        appendResult("Generation completed in " + profile.runtimeMillis() + " ms");
        appendResult("  Prompt tokens: " + result.diagnostics().inputTokenCount());
        appendResult("  Output tokens: " + result.outputTokens());
        appendResult("  Full tokens: " + result.diagnostics().fullTokenCount());
        appendResult("  Finish reason: " + result.finishReason());
        appendResult("  SmolLM2 profile runtime: " + profile.runtimeMillis() + " ms");
        appendResult("    tokenize: " + profile.tokenizeMillis() + " ms");
        appendResult("    prefill: " + profile.prefillMillis() + " ms");
        appendResult("    decoder steps: " + profile.decoderStepMillis() + " ms");
        appendResult("    lm head: " + profile.lmHeadMillis() + " ms");
        appendResult("    token select: " + profile.tokenSelectMillis() + " ms");
        appendResult("    detokenize: " + profile.detokenizeMillis() + " ms");
        appendResult("    avg/token runtime: " + profile.averageTokenRuntimeMillis(result.outputTokens()) + " ms");
        appendSmolLm2ReferenceHotspots(profile);
        List<String> decodeMicroProfile = profile.decodeMicroProfile();
        if (!decodeMicroProfile.isEmpty()) {
            appendResult("  [debug] WARP decode micro-profile (-Dsmollm2.profile.decode):");
            for (String line : decodeMicroProfile) {
                appendResult("  " + line);
            }
        }
        appendResult("  Generated token IDs: " + result.diagnostics().generatedTokenIdsPreview(32));
        List<String> stepTopK = result.diagnostics().profile().stepTopK();
        if (!stepTopK.isEmpty()) {
            appendResult("  [debug] Top-K raw logits (numerical comparison vs Transformers):");
            for (String line : stepTopK) {
                appendResult("    " + line);
            }
        }
        if (result.diagnostics().immediateEos()) {
            appendResult("  Warning: generation stopped after an immediate EOS token.");
        }
        if (result.diagnostics().emptyDecodedOutput() && result.outputTokens() > 0) {
            appendResult("  Warning: generated tokens decoded to an empty visible string.");
        }
    }


    private void appendSmolLm2ReferenceHotspots(SmolLM2GenerationProfile profile) {
        var hotspots = profile.referenceHotspots();
        if (hotspots.measuredMillis() == 0L) {
            return;
        }
        appendResult("    reference hotspots:");
        appendResult("      layer norms: " + hotspots.layerNormMillis() + " ms");
        appendResult("      attention q/k/v projections: " + hotspots.attentionProjectionMillis() + " ms");
        appendResult("      attention scores/context: " + hotspots.attentionScoreMillis() + " ms");
        appendResult("      attention output projection: " + hotspots.attentionOutputProjectionMillis() + " ms");
        appendResult("      mlp: " + hotspots.mlpMillis() + " ms");
        appendResult("      final norm: " + hotspots.finalNormMillis() + " ms");
        appendResult("      lm head: " + hotspots.lmHeadMillis() + " ms");
    }

    private void appendSmolLm2WarpReadiness(com.aresstack.windirectml.inference.smollm2.SmolLM2WarpReadinessReport report) {
        appendResult("WARP readiness: " + (report.executable() ? "executable" : "prepared, not executable"));
        appendResult("  Weight tensors: " + report.weightTensorCount()
                + ", upload: " + com.aresstack.windirectml.inference.smollm2.SmolLM2WarpReadinessReport.formatBytes(report.totalUploadBytes()));
        appendResult("  KV cache: " + com.aresstack.windirectml.inference.smollm2.SmolLM2WarpReadinessReport.formatBytes(report.totalKvCacheBytes())
                + ", scratch: " + com.aresstack.windirectml.inference.smollm2.SmolLM2WarpReadinessReport.formatBytes(report.totalScratchBytes()));
        appendResult("  Kernel steps: " + report.kernelStepCount() + ", alignment: " + report.alignmentBytes() + " bytes");
        if (report.preparedButNotExecutable()) {
            if (report.directMlProbeAvailable()) {
                appendResult("  DirectML/WARP probe available; kernel execution is the next WARP implementation step.");
            } else {
                appendResult("  Native executor missing; prepared upload/session metadata is available for the next WARP implementation step.");
            }
        }
    }

    private void runT5Generation(Path modelDir, PromptTask task, String text, int maxTokens, String selectedModel)
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
                    .task(task)
                    .userPrompt(text)
                    .maxTokens(maxTokens)
                    .temperature(0.0f)
                    .build();
            InferenceResult result = engine.generate(request, new UiTokenSink());
            if (result.getText() == null || result.getText().isBlank()) {
                appendResult("  NOTE: generated text is empty after detokenization.");
                appendResult("  Raw output tokens: " + engine.lastOutputTokenPreview());
            }
            appendResult("");
            appendResult("Generation completed in " + elapsedMs(genStart) + " ms");
            if (result.getUsage() != null) {
                appendResult("  Prompt tokens: " + result.getUsage().promptTokens());
                appendResult("  Output tokens: " + result.getUsage().completionTokens());
            }
            appendResult("  Finish reason: " + result.getFinishReason());
            for (String line : engine.lastGenerationMetrics().diagnosticLines()) {
                appendResult(line);
            }
        } finally {
            engine.shutdown();
        }
    }

    private void runQwenGeneration(Path modelDir, PromptTask task, String text, int maxTokens, String selectedModel)
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
            InferenceRequest request = InferenceRequest.builder()
                    .modelId(selectedModel)
                    .task(task)
                    .userPrompt(text)
                    .maxTokens(maxTokens)
                    .temperature(0.0f)
                    .build();
            InferenceResult result = engine.generate(request, new UiTokenSink());
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

    private final class UiTokenSink implements GenerationTokenSink {
        @Override
        public void onToken(GeneratedToken token) {
            if (token != null) {
                appendInline(token.delta());
            }
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
