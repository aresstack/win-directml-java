package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.runtime.facade.*;
import com.aresstack.windirectml.encoder.reranker.RerankResult;
import com.aresstack.windirectml.workbench.WorkbenchModel;

import javax.swing.*;
import java.awt.*;
import java.util.Arrays;
import java.util.List;

/**
 * Reranker panel: query + candidate documents, show ranked results.
 */
public final class RerankerPanel extends JPanel {

    private final WorkbenchModel model;
    private final JTextField queryField;
    private final JTextArea docsArea;
    private final JTextArea resultArea;

    public RerankerPanel(WorkbenchModel model) {
        this.model = model;
        setLayout(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        // Input panel
        var inputPanel = new JPanel(new BorderLayout(4, 4));

        var queryPanel = new JPanel(new BorderLayout(4, 4));
        queryPanel.add(new JLabel("Query:"), BorderLayout.WEST);
        queryField = new JTextField("What is Java?", 50);
        queryPanel.add(queryField, BorderLayout.CENTER);
        inputPanel.add(queryPanel, BorderLayout.NORTH);

        var docsPanel = new JPanel(new BorderLayout(4, 4));
        docsPanel.add(new JLabel("Candidate documents (one per line):"), BorderLayout.NORTH);
        docsArea = new JTextArea(6, 60);
        docsArea.setText("""
                Java is a programming language created by Sun Microsystems.
                Python is known for its simplicity and readability.
                Java 21 introduced virtual threads and pattern matching.
                C++ is a systems programming language.
                The JVM runs bytecode compiled from Java source.""");
        docsPanel.add(new JScrollPane(docsArea), BorderLayout.CENTER);
        inputPanel.add(docsPanel, BorderLayout.CENTER);

        var runBtn = new JButton("Rerank");
        runBtn.addActionListener(e -> runRerank());
        inputPanel.add(runBtn, BorderLayout.SOUTH);
        add(inputPanel, BorderLayout.NORTH);

        // Results
        resultArea = new JTextArea(10, 60);
        resultArea.setEditable(false);
        resultArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        add(new JScrollPane(resultArea), BorderLayout.CENTER);
    }

    private void runRerank() {
        String query = queryField.getText().trim();
        if (query.isEmpty()) {
            appendResult("ERROR: Query is empty.");
            return;
        }

        String[] lines = docsArea.getText().split("\n");
        List<String> documents = Arrays.stream(lines)
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .toList();

        if (documents.isEmpty()) {
            appendResult("ERROR: No candidate documents.");
            return;
        }

        appendResult("Loading reranker: " + model.getRerankerModel() + " (backend: " + model.getBackend() + ")...");

        new SwingWorker<Void, Void>() {
            @Override
            protected Void doInBackground() {
                try {
                    var config = LocalMlRuntimeConfig.builder()
                            .backend(model.getBackend())
                            .build();
                    var runtime = LocalMlRuntime.create(config);
                    var modelDir = model.getModelRoot().resolve(model.getRerankerModel());
                    var rerankConfig = new RerankerModelConfig(modelDir);

                    long startLoad = System.nanoTime();
                    try (var reranker = runtime.loadRerankerModel(rerankConfig)) {
                        long loadMs = (System.nanoTime() - startLoad) / 1_000_000;
                        appendResult("Reranker loaded in " + loadMs + " ms");

                        long startRerank = System.nanoTime();
                        List<RerankResult> results = reranker.rerank(query, documents);
                        long rerankMs = (System.nanoTime() - startRerank) / 1_000_000;

                        appendResult("Reranking completed in " + rerankMs + " ms (" + results.size() + " results)");
                        appendResult("---");
                        for (int rank = 0; rank < results.size(); rank++) {
                            RerankResult r = results.get(rank);
                            String docPreview = documents.get(r.originalIndex());
                            if (docPreview.length() > 80) docPreview = docPreview.substring(0, 77) + "...";
                            appendResult(String.format("  #%d [idx=%d] score=%.6f  %s",
                                    rank + 1, r.originalIndex(), r.score(), docPreview));
                        }
                    }
                } catch (Exception ex) {
                    appendResult("ERROR: " + ex.getClass().getSimpleName() + ": " + ex.getMessage());
                }
                return null;
            }
        }.execute();
    }

    private void appendResult(String message) {
        SwingUtilities.invokeLater(() -> {
            resultArea.append(message + "\n");
            resultArea.setCaretPosition(resultArea.getDocument().getLength());
        });
    }
}
