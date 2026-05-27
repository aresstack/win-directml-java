package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.config.generation.GenerationModelRegistry;
import com.aresstack.windirectml.config.models.EmbeddingModelRegistry;
import com.aresstack.windirectml.runtime.facade.Backend;
import com.aresstack.windirectml.workbench.WorkbenchModel;

import javax.swing.*;
import java.awt.*;
import java.awt.event.FocusAdapter;
import java.awt.event.FocusEvent;
import java.nio.file.Path;

/**
 * Configuration panel: backend selector, model root, embedding/reranker model selection, status log.
 */
public final class ConfigPanel extends JPanel {

    private final WorkbenchModel model;
    private final JComboBox<Backend> backendCombo;
    private final JTextField modelRootField;
    private final JComboBox<String> embeddingModelCombo;
    private final JComboBox<String> summarizerModelCombo;
    private final JTextField rerankerField;
    private final JTextArea logArea;

    public ConfigPanel(WorkbenchModel model) {
        this.model = model;
        setLayout(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        // Form panel
        var form = new JPanel(new GridBagLayout());
        var gbc = new GridBagConstraints();
        gbc.insets = new Insets(4, 4, 4, 4);
        gbc.anchor = GridBagConstraints.WEST;
        gbc.fill = GridBagConstraints.HORIZONTAL;

        int row = 0;

        // Backend
        gbc.gridx = 0; gbc.gridy = row; gbc.weightx = 0;
        form.add(new JLabel("Backend:"), gbc);
        backendCombo = new JComboBox<>(Backend.values());
        backendCombo.setSelectedItem(model.getBackend());
        backendCombo.addActionListener(e -> model.setBackend((Backend) backendCombo.getSelectedItem()));
        gbc.gridx = 1; gbc.weightx = 1;
        form.add(backendCombo, gbc);

        row++;

        // Model root
        gbc.gridx = 0; gbc.gridy = row; gbc.weightx = 0;
        form.add(new JLabel("Model Root:"), gbc);
        modelRootField = new JTextField(model.getModelRoot().toString(), 30);
        modelRootField.addActionListener(e -> applyModelRoot());
        modelRootField.addFocusListener(new FocusAdapter() {
            @Override
            public void focusLost(FocusEvent e) {
                applyModelRoot();
            }
        });
        gbc.gridx = 1; gbc.weightx = 1;
        form.add(modelRootField, gbc);

        row++;

        // Embedding model
        gbc.gridx = 0; gbc.gridy = row; gbc.weightx = 0;
        form.add(new JLabel("Embedding Model:"), gbc);
        embeddingModelCombo = new JComboBox<>(new String[]{
                "all-MiniLM-L6-v2",
                "e5-small-v2",
                "e5-base-v2",
                "e5-large-v2"
        });
        embeddingModelCombo.setSelectedItem(model.getEmbeddingModel());
        embeddingModelCombo.addActionListener(e ->
                model.setEmbeddingModel((String) embeddingModelCombo.getSelectedItem()));
        gbc.gridx = 1; gbc.weightx = 1;
        form.add(embeddingModelCombo, gbc);

        row++;

        // Reranker model
        gbc.gridx = 0; gbc.gridy = row; gbc.weightx = 0;
        form.add(new JLabel("Reranker Model:"), gbc);
        rerankerField = new JTextField(model.getRerankerModel(), 30);
        rerankerField.addActionListener(e -> model.setRerankerModel(rerankerField.getText().trim()));
        rerankerField.addFocusListener(new FocusAdapter() {
            @Override
            public void focusLost(FocusEvent e) {
                String value = rerankerField.getText().trim();
                if (!value.isEmpty()) {
                    model.setRerankerModel(value);
                }
            }
        });
        gbc.gridx = 1; gbc.weightx = 1;
        form.add(rerankerField, gbc);

        row++;

        // Summarizer model (uses GenerationModelRegistry for generation-capable models)
        gbc.gridx = 0; gbc.gridy = row; gbc.weightx = 0;
        form.add(new JLabel("Summarizer Model:"), gbc);
        var summarizerOptions = GenerationModelRegistry.runnableEntries()
                .stream().map(GenerationModelRegistry.Entry::modelId).toArray(String[]::new);
        summarizerModelCombo = new JComboBox<>(summarizerOptions);
        summarizerModelCombo.setSelectedItem(model.getSummarizerModel());
        summarizerModelCombo.addActionListener(e ->
                model.setSummarizerModel((String) summarizerModelCombo.getSelectedItem()));
        gbc.gridx = 1; gbc.weightx = 1;
        form.add(summarizerModelCombo, gbc);

        add(form, BorderLayout.NORTH);

        // Status / log area
        logArea = new JTextArea(12, 60);
        logArea.setEditable(false);
        logArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        add(new JScrollPane(logArea), BorderLayout.CENTER);

        appendLog("DirectML Workbench initialized.");
        appendLog("Backend: " + model.getBackend());
        appendLog("Model root: " + model.getModelRoot());
    }

    private void applyModelRoot() {
        String value = modelRootField.getText().trim();
        if (!value.isEmpty()) {
            model.setModelRoot(Path.of(value));
        }
    }

    public void appendLog(String message) {
        SwingUtilities.invokeLater(() -> {
            logArea.append(message + "\n");
            logArea.setCaretPosition(logArea.getDocument().getLength());
        });
    }

    public JTextArea getLogArea() { return logArea; }
}
