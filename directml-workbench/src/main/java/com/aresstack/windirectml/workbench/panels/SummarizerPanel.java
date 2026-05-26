package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.encoder.e5.E5Variant;
import com.aresstack.windirectml.runtime.facade.EmbeddingModelConfig;
import com.aresstack.windirectml.runtime.facade.LocalMlRuntime;
import com.aresstack.windirectml.runtime.facade.LocalMlRuntimeConfig;
import com.aresstack.windirectml.workbench.WorkbenchModel;

import javax.swing.*;
import java.awt.*;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Extractive summarizer demo built on the public embedding API.
 * <p>
 * This is intentionally not a decoder/LLM summarizer. It embeds the whole text
 * and individual sentences, then returns the sentences closest to the document
 * vector in original order.
 */
public final class SummarizerPanel extends JPanel {

    private final WorkbenchModel model;
    private final JTextArea inputArea;
    private final JTextArea resultArea;
    private final JSpinner sentenceCountSpinner;

    public SummarizerPanel(WorkbenchModel model) {
        this.model = model;
        setLayout(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        var inputPanel = new JPanel(new BorderLayout(4, 4));
        inputPanel.add(new JLabel("Text to summarize:"), BorderLayout.NORTH);
        inputArea = new JTextArea(10, 70);
        inputArea.setLineWrap(true);
        inputArea.setWrapStyleWord(true);
        inputArea.setText("Paste a longer text here. The workbench will create an extractive summary by selecting the most representative sentences using the selected embedding model.");
        inputPanel.add(new JScrollPane(inputArea), BorderLayout.CENTER);

        var controls = new JPanel(new FlowLayout(FlowLayout.LEFT));
        controls.add(new JLabel("Sentences:"));
        sentenceCountSpinner = new JSpinner(new SpinnerNumberModel(3, 1, 20, 1));
        controls.add(sentenceCountSpinner);
        var runBtn = new JButton("Summarize");
        runBtn.addActionListener(e -> runSummarizer());
        controls.add(runBtn);
        inputPanel.add(controls, BorderLayout.SOUTH);
        add(inputPanel, BorderLayout.NORTH);

        resultArea = new JTextArea(14, 70);
        resultArea.setEditable(false);
        resultArea.setLineWrap(true);
        resultArea.setWrapStyleWord(true);
        resultArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        add(new JScrollPane(resultArea), BorderLayout.CENTER);
    }

    private void runSummarizer() {
        String text = inputArea.getText().trim();
        if (text.isEmpty()) {
            appendResult("ERROR: Input text is empty.");
            return;
        }

        List<String> sentences = splitSentences(text);
        if (sentences.isEmpty()) {
            appendResult("ERROR: No sentences found.");
            return;
        }

        int requested = (Integer) sentenceCountSpinner.getValue();
        int count = Math.min(requested, sentences.size());
        appendResult("Loading model: " + model.getEmbeddingModel() + " (backend: " + model.getBackend() + ")...");

        new SwingWorker<Void, Void>() {
            @Override
            protected Void doInBackground() {
                try {
                    var config = LocalMlRuntimeConfig.builder()
                            .backend(model.getBackend())
                            .build();
                    var runtime = LocalMlRuntime.create(config);
                    Path modelDir = model.getModelRoot().resolve(model.getEmbeddingModel());

                    var embedConfig = buildEmbeddingConfig(modelDir);
                    long start = System.nanoTime();
                    try (var embeddings = runtime.loadEmbeddingModel(embedConfig)) {
                        appendResult("Model loaded in " + elapsedMs(start) + " ms (dimension: " + embeddings.dimension() + ")");

                        long summarizeStart = System.nanoTime();
                        float[] documentVector = embeddings.embed(text);
                        List<float[]> sentenceVectors = embeddings.embedBatch(sentences);

                        List<ScoredSentence> ranked = new ArrayList<>();
                        for (int i = 0; i < sentences.size(); i++) {
                            ranked.add(new ScoredSentence(i, sentences.get(i), cosine(documentVector, sentenceVectors.get(i))));
                        }

                        ranked.sort(Comparator.comparingDouble(ScoredSentence::score).reversed());
                        List<ScoredSentence> selected = ranked.subList(0, count).stream()
                                .sorted(Comparator.comparingInt(ScoredSentence::index))
                                .toList();

                        appendResult("Summary completed in " + elapsedMs(summarizeStart) + " ms");
                        appendResult("Selected " + selected.size() + " of " + sentences.size() + " sentences.");
                        appendResult("");
                        appendResult("SUMMARY:");
                        for (ScoredSentence sentence : selected) {
                            appendResult("- " + sentence.text());
                        }
                        appendResult("");
                        appendResult("Scores:");
                        for (ScoredSentence sentence : selected) {
                            appendResult("  #" + (sentence.index() + 1) + " score=" + sentence.score());
                        }
                    }
                } catch (Exception ex) {
                    appendResult("ERROR: " + ex.getClass().getSimpleName() + ": " + ex.getMessage());
                }
                return null;
            }
        }.execute();
    }

    private EmbeddingModelConfig buildEmbeddingConfig(Path modelDir) {
        String selected = model.getEmbeddingModel();
        return switch (selected) {
            case "e5-small-v2" -> EmbeddingModelConfig.e5(modelDir, E5Variant.SMALL_V2, "query: ");
            case "e5-base-v2" -> EmbeddingModelConfig.e5(modelDir, E5Variant.BASE_V2, "query: ");
            case "e5-large-v2" -> EmbeddingModelConfig.e5(modelDir, E5Variant.LARGE_V2, "query: ");
            default -> EmbeddingModelConfig.miniLm(modelDir);
        };
    }

    private static List<String> splitSentences(String text) {
        String normalized = text.replace('\n', ' ').replace('\r', ' ').trim();
        String[] parts = normalized.split("(?<=[.!?])\\s+");
        List<String> sentences = new ArrayList<>();
        for (String part : parts) {
            String sentence = part.trim();
            if (!sentence.isEmpty()) {
                sentences.add(sentence);
            }
        }
        return sentences;
    }

    private static double cosine(float[] a, float[] b) {
        double dot = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        int n = Math.min(a.length, b.length);
        for (int i = 0; i < n; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        if (normA == 0.0 || normB == 0.0) {
            return 0.0;
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
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

    private record ScoredSentence(int index, String text, double score) { }
}
