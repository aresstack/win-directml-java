package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.runtime.facade.*;
import com.aresstack.windirectml.encoder.e5.E5Variant;
import com.aresstack.windirectml.workbench.WorkbenchModel;

import javax.swing.*;
import java.awt.*;
import java.util.Arrays;

/**
 * Single-text embedding panel: load model, embed text, show results.
 */
public final class EmbeddingsPanel extends JPanel {

    private final WorkbenchModel model;
    private final JTextField inputField;
    private final JTextArea resultArea;

    public EmbeddingsPanel(WorkbenchModel model) {
        this.model = model;
        setLayout(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        // Input area
        var inputPanel = new JPanel(new BorderLayout(4, 4));
        inputPanel.add(new JLabel("Text to embed:"), BorderLayout.NORTH);
        inputField = new JTextField(60);
        inputField.setText("Hello, this is a test sentence for embedding.");
        inputPanel.add(inputField, BorderLayout.CENTER);

        var runBtn = new JButton("Embed");
        runBtn.addActionListener(e -> runEmbed());
        inputPanel.add(runBtn, BorderLayout.EAST);
        add(inputPanel, BorderLayout.NORTH);

        // Result area
        resultArea = new JTextArea(18, 60);
        resultArea.setEditable(false);
        resultArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        add(new JScrollPane(resultArea), BorderLayout.CENTER);
    }

    private void runEmbed() {
        String text = inputField.getText().trim();
        if (text.isEmpty()) {
            appendResult("ERROR: Input text is empty.");
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
                        float[] vector = embeddings.embed(text);
                        long embedMs = (System.nanoTime() - startEmbed) / 1_000_000;

                        appendResult("Embedding completed in " + embedMs + " ms");
                        appendResult("Dimension: " + vector.length);
                        int preview = Math.min(10, vector.length);
                        appendResult("First " + preview + " values: " +
                                Arrays.toString(Arrays.copyOf(vector, preview)));
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
