package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.runtime.facade.*;
import com.aresstack.windirectml.encoder.e5.E5Variant;
import com.aresstack.windirectml.workbench.WorkbenchModel;

import javax.swing.*;
import java.awt.*;
import java.util.Arrays;
import java.util.List;

/**
 * Batch embeddings panel: multiline input, embedBatch, show results.
 */
public final class BatchEmbeddingsPanel extends JPanel {

    private final WorkbenchModel model;
    private final JTextArea inputArea;
    private final JTextArea resultArea;

    public BatchEmbeddingsPanel(WorkbenchModel model) {
        this.model = model;
        setLayout(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        // Input
        var inputPanel = new JPanel(new BorderLayout(4, 4));
        inputPanel.add(new JLabel("Texts (one per line):"), BorderLayout.NORTH);
        inputArea = new JTextArea(6, 60);
        inputArea.setText("The quick brown fox jumps over the lazy dog.\nJava 21 enables modern FFM-based inference.\nDirectML accelerates ML workloads on Windows.");
        inputPanel.add(new JScrollPane(inputArea), BorderLayout.CENTER);

        var runBtn = new JButton("Embed Batch");
        runBtn.addActionListener(e -> runBatch());
        inputPanel.add(runBtn, BorderLayout.SOUTH);
        add(inputPanel, BorderLayout.NORTH);

        // Result
        resultArea = new JTextArea(12, 60);
        resultArea.setEditable(false);
        resultArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        add(new JScrollPane(resultArea), BorderLayout.CENTER);
    }

    private void runBatch() {
        String[] lines = inputArea.getText().split("\n");
        List<String> texts = Arrays.stream(lines)
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .toList();

        if (texts.isEmpty()) {
            appendResult("ERROR: No input texts.");
            return;
        }

        appendResult("Loading model: " + model.getEmbeddingModel() + " (backend: " + model.getBackend() + ")...");

        new SwingWorker<Void, Void>() {
            @Override
            protected Void doInBackground() {
                try {
                    var config = LocalMlRuntimeConfig.builder()
                            .backend(model.getBackend())
                            .build();
                    var runtime = LocalMlRuntime.create(config);
                    var modelDir = model.getModelRoot().resolve(model.getEmbeddingModel());

                    var embedConfig = buildEmbeddingConfig(modelDir);
                    long startLoad = System.nanoTime();
                    try (var embeddings = runtime.loadEmbeddingModel(embedConfig)) {
                        long loadMs = (System.nanoTime() - startLoad) / 1_000_000;
                        appendResult("Model loaded in " + loadMs + " ms (dimension: " + embeddings.dimension() + ")");

                        long startEmbed = System.nanoTime();
                        List<float[]> results = embeddings.embedBatch(texts);
                        long embedMs = (System.nanoTime() - startEmbed) / 1_000_000;

                        appendResult("Batch embedding completed in " + embedMs + " ms");
                        appendResult("Count: " + results.size() + ", Dimension: " + (results.isEmpty() ? 0 : results.getFirst().length));
                        for (int i = 0; i < results.size(); i++) {
                            float[] v = results.get(i);
                            int preview = Math.min(5, v.length);
                            appendResult("  [" + i + "] first " + preview + ": " +
                                    Arrays.toString(Arrays.copyOf(v, preview)) + "...");
                        }
                    }
                } catch (Exception ex) {
                    appendResult("ERROR: " + ex.getClass().getSimpleName() + ": " + ex.getMessage());
                }
                return null;
            }
        }.execute();
    }

    private EmbeddingModelConfig buildEmbeddingConfig(java.nio.file.Path modelDir) {
        String selected = model.getEmbeddingModel();
        return switch (selected) {
            case "e5-small-v2" -> EmbeddingModelConfig.e5(modelDir, E5Variant.SMALL_V2, "query: ");
            case "e5-base-v2" -> EmbeddingModelConfig.e5(modelDir, E5Variant.BASE_V2, "query: ");
            case "e5-large-v2" -> EmbeddingModelConfig.e5(modelDir, E5Variant.LARGE_V2, "query: ");
            default -> EmbeddingModelConfig.miniLm(modelDir);
        };
    }

    private void appendResult(String message) {
        SwingUtilities.invokeLater(() -> {
            resultArea.append(message + "\n");
            resultArea.setCaretPosition(resultArea.getDocument().getLength());
        });
    }
}
