package com.aresstack.windirectml.sidecar.workbench.panels;

import com.aresstack.windirectml.sidecar.client.SidecarClientConfig;
import com.aresstack.windirectml.sidecar.client.SidecarException;
import com.aresstack.windirectml.sidecar.protocol.validation.ModelValidator;
import com.aresstack.windirectml.sidecar.protocol.validation.ValidationReport;
import com.aresstack.windirectml.sidecar.workbench.WorkbenchModel;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.SwingWorker;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

/**
 * Configuration form + process control buttons.
 */
public final class ConfigPanel extends JPanel {

    private static final long serialVersionUID = 1L;

    private final WorkbenchModel model;

    private final JTextField javaExeField = new JTextField(System.getProperty("java.home")
            + java.io.File.separator + "bin" + java.io.File.separator + "java");
    private final JTextField jarPathField = new JTextField(
            "directml-sidecar/build/libs/directml-sidecar-0.1.0-SNAPSHOT-all.jar");
    private final JComboBox<String> embedModelBox =
            new JComboBox<String>(EmbedModelOptions.embeddingOptions().toArray(new String[0]));
    private final JTextField modelDirField = new JTextField("model/all-MiniLM-L6-v2");
    private final JComboBox<String> e5VariantBox =
            new JComboBox<String>(new String[]{
                    "base-sts-en-de", "small-v2", "base-v2", "large-v2"});
    private final JTextField e5ModelDirField = new JTextField("");
    private final JComboBox<String> backendBox =
            new JComboBox<String>(new String[]{"auto", "directml", "cpu"});
    private final JTextField rerankModelDirField = new JTextField("");
    private final JComboBox<String> rerankBackendBox =
            new JComboBox<String>(new String[]{"auto", "directml", "cpu"});
    private final JTextField phi3ModelDirField = new JTextField(
            "model/phi3-mini-directml-int4/directml/directml-int4-awq-block-128");
    private final JComboBox<String> phi3BackendBox =
            new JComboBox<String>(new String[]{"auto", "directml", "cpu"});
    private final JTextField phi3MaxTokensField = new JTextField("512");
    private final JCheckBox debugBox = new JCheckBox("windirectml.debug=true");
    private final JTextField dllOverrideField = new JTextField("");
    private final JTextField timeoutField = new JTextField("30000");
    private final JTextField summarizeTimeoutField = new JTextField("300000");
    private final JTextField extraJvmField = new JTextField("-Xmx8g");

    private final JTextArea commandPreview = new JTextArea(4, 60);

    private final JButton startBtn = new JButton("Start Sidecar");
    private final JButton stopBtn = new JButton("Stop Sidecar");
    private final JButton restartBtn = new JButton("Restart Sidecar");
    private final JButton healthBtn = new JButton("Health");
    private final JButton validateBtn = new JButton("Validate Models");
    private final JButton clearBtn = new JButton("Clear Logs");

    private final JTextArea logArea = new JTextArea(8, 60);

    private Runnable onChange;

    public ConfigPanel(WorkbenchModel model) {
        this.model = model;
        setLayout(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

        JPanel form = new JPanel(new GridBagLayout());
        GridBagConstraints c = new GridBagConstraints();
        c.fill = GridBagConstraints.HORIZONTAL;
        c.insets = new Insets(3, 3, 3, 3);
        int row = 0;
        addRow(form, c, row++, "Java executable", javaExeField);
        addRow(form, c, row++, "Sidecar jar path", jarPathField);
        addRow(form, c, row++, "embed.model", embedModelBox);
        addRow(form, c, row++, "MiniLM directory", modelDirField);
        addRow(form, c, row++, "E5 variant", e5VariantBox);
        addRow(form, c, row++, "E5 directory (optional)", e5ModelDirField);
        addRow(form, c, row++, "embed.backend", backendBox);
        addRow(form, c, row++, "Reranker directory (optional)", rerankModelDirField);
        addRow(form, c, row++, "rerank.backend", rerankBackendBox);
        addRow(form, c, row++, "Phi-3 directory (summarizer)", phi3ModelDirField);
        addRow(form, c, row++, "phi3.backend", phi3BackendBox);
        addRow(form, c, row++, "phi3.maxTokens", phi3MaxTokensField);
        addRow(form, c, row++, "DirectML.dll override", dllOverrideField);
        addRow(form, c, row++, "Extra JVM args", extraJvmField);
        addRow(form, c, row++, "Request timeout (ms)", timeoutField);
        addRow(form, c, row++, "Summarize timeout (ms)", summarizeTimeoutField);
        addRow(form, c, row++, "Debug", debugBox);

        JPanel buttons = new JPanel();
        buttons.add(startBtn);
        buttons.add(stopBtn);
        buttons.add(restartBtn);
        buttons.add(Box.createHorizontalStrut(20));
        buttons.add(healthBtn);
        buttons.add(validateBtn);
        buttons.add(clearBtn);

        commandPreview.setEditable(false);
        commandPreview.setLineWrap(true);
        commandPreview.setWrapStyleWord(false);
        commandPreview.setFont(logArea.getFont().deriveFont((float) (logArea.getFont().getSize() - 1)));

        logArea.setEditable(false);
        logArea.setBackground(new Color(248, 248, 248));

        JPanel north = new JPanel(new BorderLayout(0, 6));
        north.add(form, BorderLayout.CENTER);
        north.add(buttons, BorderLayout.SOUTH);

        JPanel south = new JPanel(new BorderLayout(0, 6));
        south.add(new JScrollPane(commandPreview), BorderLayout.NORTH);
        south.add(new JScrollPane(logArea), BorderLayout.CENTER);

        add(north, BorderLayout.NORTH);
        add(south, BorderLayout.CENTER);

        wireActions();
        applyConfigToModel();
        refreshCommandPreview();
    }

    public void setOnChange(Runnable r) {
        this.onChange = r;
    }

    private void addRow(JPanel form, GridBagConstraints c, int row, String label, java.awt.Component field) {
        c.gridx = 0;
        c.gridy = row;
        c.weightx = 0;
        form.add(new JLabel(label), c);
        c.gridx = 1;
        c.weightx = 1;
        form.add(field, c);
    }

    private void wireActions() {
        ActionListener applyAndPreview = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                applyConfigToModel();
                refreshCommandPreview();
            }
        };
        backendBox.addActionListener(applyAndPreview);
        rerankBackendBox.addActionListener(applyAndPreview);
        phi3BackendBox.addActionListener(applyAndPreview);
        debugBox.addActionListener(applyAndPreview);
        embedModelBox.addActionListener(applyAndPreview);
        e5VariantBox.addActionListener(applyAndPreview);

        startBtn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                applyConfigToModel();
                refreshCommandPreview();
                runAsync("Start Sidecar", new Callable() {
                    @Override
                    public String call() throws SidecarException {
                        model.startSidecar();
                        return "Sidecar started: pid not exposed by Java 8 API";
                    }
                });
            }
        });
        stopBtn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                runAsync("Stop Sidecar", new Callable() {
                    @Override
                    public String call() {
                        model.stopSidecar();
                        return model.lastStopInfo().describe();
                    }
                });
            }
        });
        restartBtn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                applyConfigToModel();
                refreshCommandPreview();
                runAsync("Restart Sidecar", new Callable() {
                    @Override
                    public String call() throws SidecarException {
                        model.restartSidecar();
                        return "Sidecar restarted";
                    }
                });
            }
        });
        healthBtn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                runAsync("Health", new Callable() {
                    @Override
                    public String call() throws SidecarException {
                        return "health -> " + model.health().getStatus()
                                + " (embed=" + model.health().getEmbeddingBackend()
                                + ", embedReady=" + model.health().isEmbeddingReady()
                                + ", summarizer=" + model.health().getSummarizerBackend()
                                + ", summarizerReady=" + model.health().isSummarizerReady() + ")";
                    }
                });
            }
        });
        validateBtn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                applyConfigToModel();
                runAsync("Validate Models", new Callable() {
                    @Override
                    public String call() {
                        return validateConfiguredModels();
                    }
                });
            }
        });
        clearBtn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                logArea.setText("");
            }
        });
    }

    private String validateConfiguredModels() {
        StringBuilder sb = new StringBuilder();
        String embedModel = (String) embedModelBox.getSelectedItem();
        if (embedModel != null && embedModel.toLowerCase(java.util.Locale.ROOT).contains("e5")) {
            sb.append(ModelValidator.validate(fileOrNull(e5ModelDirField.getText()),
                    ModelValidator.e5Expectation((String) e5VariantBox.getSelectedItem())).format());
        } else {
            sb.append(ModelValidator.validate(fileOrNull(modelDirField.getText()),
                    ModelValidator.minilmExpectation()).format());
        }
        String rerankDir = rerankModelDirField.getText();
        if (rerankDir != null && rerankDir.trim().length() > 0) {
            sb.append('\n').append('\n');
            sb.append(ModelValidator.validate(fileOrNull(rerankDir),
                    ModelValidator.rerankerExpectation()).format());
        }
        return sb.toString();
    }

    private static File fileOrNull(String value) {
        if (value == null || value.trim().length() == 0) return null;
        return new File(value.trim());
    }

    private void applyConfigToModel() {
        SidecarClientConfig cfg = model.getConfig();
        cfg.setJavaExecutable(javaExeField.getText().trim());
        cfg.setSidecarJarPath(jarPathField.getText().trim());
        cfg.setModelDirectory(modelDirField.getText().trim());
        cfg.setEmbedModel((String) embedModelBox.getSelectedItem());
        cfg.setE5Variant((String) e5VariantBox.getSelectedItem());
        cfg.setE5ModelDirectory(e5ModelDirField.getText().trim());
        cfg.setEmbedBackend((String) backendBox.getSelectedItem());
        cfg.setRerankModelDirectory(rerankModelDirField.getText().trim());
        cfg.setRerankBackend((String) rerankBackendBox.getSelectedItem());
        cfg.setPhi3ModelDirectory(phi3ModelDirField.getText().trim());
        cfg.setPhi3Backend((String) phi3BackendBox.getSelectedItem());
        try {
            int mt = Integer.parseInt(phi3MaxTokensField.getText().trim());
            cfg.setPhi3MaxTokens(mt > 0 ? mt : 0);
        } catch (NumberFormatException ignored) {
        }
        cfg.setDirectmlDebug(debugBox.isSelected());
        cfg.setDirectmlDllOverride(dllOverrideField.getText().trim());
        cfg.setExtraJvmArgs(extraJvmField.getText().trim());
        try {
            long t = Long.parseLong(timeoutField.getText().trim());
            if (t > 0) cfg.setRequestTimeoutMillis(t);
        } catch (NumberFormatException ignored) {
        }
        try {
            long st = Long.parseLong(summarizeTimeoutField.getText().trim());
            if (st > 0) cfg.setSummarizeTimeoutMillis(st);
        } catch (NumberFormatException ignored) {
        }
    }

    private void refreshCommandPreview() {
        java.util.List<String> cmd = model.getCommandLine();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < cmd.size(); i++) {
            if (i > 0) sb.append(' ');
            String s = cmd.get(i);
            if (s.contains(" ")) sb.append('"').append(s).append('"');
            else sb.append(s);
        }
        commandPreview.setText(sb.toString());
        commandPreview.setCaretPosition(0);
    }

    private interface Callable {
        String call() throws Exception;
    }

    private void runAsync(final String action, final Callable c) {
        appendLog("-> " + action + "...");
        SwingWorker<String, Void> w = new SwingWorker<String, Void>() {
            @Override
            protected String doInBackground() throws Exception {
                return c.call();
            }

            @Override
            protected void done() {
                try {
                    appendLog(get());
                } catch (Exception e) {
                    Throwable cause = (e.getCause() != null) ? e.getCause() : e;
                    appendLog(action + " failed: " + cause.getMessage());
                }
                if (onChange != null) onChange.run();
            }
        };
        w.execute();
    }

    private void appendLog(String msg) {
        logArea.append(msg);
        logArea.append("\n");
        logArea.setCaretPosition(logArea.getDocument().getLength());
    }

    @Override
    public Dimension getPreferredSize() {
        Dimension d = super.getPreferredSize();
        d.width = Math.max(d.width, 800);
        return d;
    }
}
