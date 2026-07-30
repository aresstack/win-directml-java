package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.config.generation.GenerationModelRegistry;
import com.aresstack.windirectml.config.generation.GenerationModelRegistry.Entry;
import com.aresstack.windirectml.config.generation.GenerationOutputMode;
import com.aresstack.windirectml.inference.prompt.PromptStrategies;
import com.aresstack.windirectml.inference.prompt.PromptTask;
import com.aresstack.windirectml.catalog.CatalogBackend;
import com.aresstack.windirectml.inference.api.GenerationException;
import com.aresstack.windirectml.inference.api.GenerationResult;
import com.aresstack.windirectml.inference.api.GenerationTokenListener;
import com.aresstack.windirectml.runtime.facade.Backend;
import com.aresstack.windirectml.workbench.WorkbenchModel;
import com.aresstack.windirectml.workbench.artifact.ModelRuntimeRegistry;
import com.aresstack.windirectml.workbench.runtime.WorkbenchGenerationService;
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
 * The Workbench is the manual test surface for decoder runtimes. All generation is dispatched through
 * the shared, neutral {@link WorkbenchGenerationService} → {@code GenerationRuntime} (the exact runtime
 * AskAI uses). The panel keeps no per-family orchestration and no Python bridge: it resolves the selected
 * model against the neutral catalog, applies the chosen {@link PromptTask} via the shared
 * {@link PromptStrategies} pipeline, and lets the runtime enforce the backend matrix and package-only load.
 */
public final class SummarizerPanel extends JPanel {

    private static final String QWEN05_MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct";
    private static final String GEMMA3_MODEL_ID_PREFIX = "google/gemma-3-";
    private static final String SMOLLM2_MODEL_ID_PREFIX = "HuggingFaceTB/SmolLM2-";

    private final WorkbenchModel model;
    private final ModelRuntimeRegistry runtimeRegistry;
    // The shared, neutral generation runtime (same one AskAI uses). The panel routes generation
    // through this service instead of its own per-family orchestration (W4).
    private final WorkbenchGenerationService generationService = new WorkbenchGenerationService();
    private final JTextArea inputArea;
    private final JTextArea resultArea;
    private final JComboBox<String> modelSelector;
    private final JComboBox<PromptTask> promptTemplateSelector;
    private final JSpinner maxTokensSpinner;
    private final JCheckBox streamingCheckbox;

    public SummarizerPanel(WorkbenchModel model) {
        this.model = model;
        this.runtimeRegistry = new ModelRuntimeRegistry(model);
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

        streamingCheckbox = new JCheckBox("Streaming output",
                GenerationOutputMode.fromSystemProperty().isStreaming());
        streamingCheckbox.setToolTipText("Show tokens live as they are generated. Uncheck to buffer and "
                + "show the full result at the end. Default and initial value come from "
                + "-Ddirectml.generation.output / -Ddirectml.generation.streaming.");
        runControls.add(streamingCheckbox);
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
        boolean gemma3Model = isGemma3Model(selectedModel);
        boolean smolLm2Model = isSmolLm2Model(selectedModel);
        Entry entry = GenerationModelRegistry.findByModelId(selectedModel);
        if (entry != null) {
            if (entry.status() == GenerationModelRegistry.Status.UNSUPPORTED) {
                appendResult("ERROR: Model '" + selectedModel + "' is unsupported in this runtime.");
                return;
            }
            // Runnable models (SHIPPED/EXPERIMENTAL) are never blocked here; only genuinely PLANNED models are.
            // No hard-coded family exemptions for status-runnable models (Qwen 0.5B: STATUS-2; SmolLM2:
            // SMOLLM2-PRODUCT-AUDIT-1 — both EXPERIMENTAL). gemma3Model stays: the Gemma *base* checkpoint is
            // PLANNED but routes to the Gemma handler for a clear missing-package message.
            if (entry.status() == GenerationModelRegistry.Status.PLANNED && !gemma3Model) {
                appendResult("ERROR: Model '" + selectedModel + "' is selectable but not executable yet.");
                appendResult("  Status: planned. Runtime support is in progress for family "
                        + entry.architecture().token() + ".");
                return;
            }
        }

        int maxTokens = (Integer) maxTokensSpinner.getValue();
        boolean streaming = streamingCheckbox.isSelected();
        appendResult("Loading generation model: " + selectedModel
                + " (backend: " + model.getBackend() + ", maxTokens: " + maxTokens
                + ", output: " + (streaming ? "streaming" : "buffered") + ")...");
        if (qwenTestModel) {
            appendResult("  NOTE: Qwen acceleration depends on WARP/AUTO and the selected package source (see Config/Download tabs).");
        } else if (gemma3Model) {
            appendResult("  NOTE: Gemma 3 runs the native Java/DirectML runtime on WARP/AUTO; weights load from the "
                    + "compiled model_gemma3.wdmlpack. No Python. CPU is not offered for Gemma — the runtime rejects "
                    + "it as UNSUPPORTED_BACKEND.");
        } else if (smolLm2Model) {
            appendResult("  NOTE: SmolLM2 runs on AUTO or CPU. AUTO uses hardware DirectML when available and may "
                    + "resolve to the CPU reference runtime; explicit CPU uses the reference runtime. Software WARP "
                    + "is currently unsupported (problems.md P1).");
            appendResult("  NOTE: Requires a prebuilt model.wdmlpack. Use the Download tab -> Convert; inference never compiles.");
        } else if (isT5Model(selectedModel)) {
            appendResult("  NOTE: T5-family models run only from a prebuilt .wdmlpack. Use the Download tab -> Convert; inference never compiles.");
            appendResult("  NOTE: Backend = CPU runs the validated Java reference seq2seq runtime. Backend = WARP/AUTO routes the "
                    + "dense projections (attention/feed-forward + LM-head matmuls) through DirectML on the WARP software / "
                    + "hardware adapter, while layer norms, attention softmax, and relative-position bias stay on the CPU "
                    + "reference path. All curated T5 models (google-t5/t5-small, google/flan-t5-small, "
                    + "Salesforce/codet5-small, Salesforce/codet5-base-multi-sum) are correctness-certified (CPU == WARP, "
                    + "greedy; T5-REALMODEL-CERT-1..4). No Python on any T5 path.");
        }

        new SwingWorker<Void, Void>() {
            @Override
            protected Void doInBackground() {
                // W4: single dispatch through the shared neutral GenerationRuntime — no per-family
                // orchestration, no Python. Model resolution is catalog-driven; the runtime enforces
                // the backend matrix and loads package-only. The selected PromptTask is applied by the
                // service via the shared family-aware PromptStrategies (P7). (Qwen keeps the shared
                // descriptor's runtime dir so the status line and the load path agree.)
                try {
                    Path modelDir = qwenTestModel
                            ? runtimeRegistry.qwen05bRuntimeDir()
                            : resolveSummarizerModelDir(selectedModel);
                    CatalogBackend backend = WorkbenchGenerationService.toCatalogBackend(model.getBackend());
                    GenerationTokenListener listener = streaming ? token -> appendInline(token.text()) : null;
                    GenerationResult result = generationService.generate(
                            selectedModel, modelDir, backend, promptTask, text, maxTokens, listener);
                    if (streaming) {
                        appendResult(""); // newline after streamed text
                    } else {
                        appendResult(result.text());
                    }
                    appendResult("");
                    appendResult("Backend: " + result.backend() + " | finish: " + result.finishReason()
                            + " | prompt tokens: " + result.promptTokenCount()
                            + " | output tokens: " + result.generatedTokenCount());
                } catch (GenerationException ex) {
                    appendResult("GENERATION ERROR [" + ex.errorCode() + "]: " + ex.getMessage());
                } catch (Exception ex) {
                    appendResult("ERROR: " + ex.getClass().getSimpleName() + ": " + ex.getMessage());
                }
                return null;
            }
        }.execute();
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

    private static boolean isQwenTestModel(String modelId) {
        return QWEN05_MODEL_ID.equals(modelId);
    }

    private static boolean isGemma3Model(String modelId) {
        return modelId != null && modelId.startsWith(GEMMA3_MODEL_ID_PREFIX);
    }

    private static boolean isSmolLm2Model(String modelId) {
        return modelId != null && modelId.startsWith(SMOLLM2_MODEL_ID_PREFIX);
    }

    private static boolean isT5Model(String modelId) {
        Entry entry = GenerationModelRegistry.findByModelId(modelId);
        return entry != null && entry.architecture() == GenerationModelRegistry.Architecture.SEQ2SEQ;
    }

    private void appendResult(String message) {
        SwingUtilities.invokeLater(() -> {
            resultArea.append(message + "\n");
            resultArea.setCaretPosition(resultArea.getDocument().getLength());
        });
    }
}
