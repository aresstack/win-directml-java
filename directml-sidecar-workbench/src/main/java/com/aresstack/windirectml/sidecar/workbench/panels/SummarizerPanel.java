package com.aresstack.windirectml.sidecar.workbench.panels;

import com.aresstack.windirectml.sidecar.client.JsonRpcError;
import com.aresstack.windirectml.sidecar.client.SummaryResult;
import com.aresstack.windirectml.sidecar.workbench.WorkbenchModel;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSpinner;
import javax.swing.JTextArea;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingWorker;

import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public final class SummarizerPanel extends JPanel {

    private static final long serialVersionUID = 1L;

    private final WorkbenchModel model;
    private final JTextArea inputArea = new JTextArea(8, 60);
    private final JSpinner maxTokensSpinner = new JSpinner(new SpinnerNumberModel(256, 16, 4096, 16));
    private final JButton  summarizeBtn = new JButton("Summarize");
    private final JLabel   timingLbl = new JLabel("timing: —");
    private final JTextArea outputArea = new JTextArea(8, 60);
    private final JTextArea rawArea = new JTextArea(6, 60);

    public SummarizerPanel(WorkbenchModel model) {
        this.model = model;
        setLayout(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

        inputArea.setText("Paste a paragraph here and click Summarize.");

        JPanel controls = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 0));
        controls.add(new JLabel("maxTokens:"));
        controls.add(maxTokensSpinner);
        controls.add(summarizeBtn);
        controls.add(timingLbl);

        outputArea.setEditable(false);
        outputArea.setLineWrap(true);
        outputArea.setWrapStyleWord(true);
        rawArea.setEditable(false);
        rawArea.setLineWrap(true);

        JScrollPane in  = new JScrollPane(inputArea);
        in.setBorder(BorderFactory.createTitledBorder("Input text"));
        JScrollPane out = new JScrollPane(outputArea);
        out.setBorder(BorderFactory.createTitledBorder("Summary output"));
        JScrollPane raw = new JScrollPane(rawArea);
        raw.setBorder(BorderFactory.createTitledBorder("Raw JSON-RPC response"));

        JPanel center = new JPanel(new java.awt.GridLayout(3, 1, 0, 6));
        center.add(in);
        center.add(out);
        center.add(raw);

        add(controls, BorderLayout.NORTH);
        add(center, BorderLayout.CENTER);

        summarizeBtn.addActionListener(new ActionListener() {
            @Override public void actionPerformed(ActionEvent e) { runSummarize(); }
        });
    }

    private void runSummarize() {
        if (!model.isRunning()) {
            outputArea.setText("sidecar not running");
            return;
        }
        final String text = inputArea.getText();
        final int maxTokens = ((Number) maxTokensSpinner.getValue()).intValue();
        timingLbl.setText("timing: …");
        outputArea.setText("");
        rawArea.setText("");
        new SwingWorker<SummaryResult, Void>() {
            @Override protected SummaryResult doInBackground() throws Exception {
                return model.summarize(text, maxTokens);
            }
            @Override protected void done() {
                try {
                    SummaryResult r = get();
                    outputArea.setText(r.getText() == null ? "" : r.getText());
                    rawArea.setText(r.getRaw() == null ? "" : r.getRaw());
                    timingLbl.setText("timing: " + r.getElapsedMillis() + " ms ("
                            + r.getPromptTokens() + "→" + r.getOutputTokens()
                            + ", finish=" + r.getFinishReason() + ")");
                } catch (Exception ex) {
                    Throwable cause = (ex.getCause() != null) ? ex.getCause() : ex;
                    if (cause instanceof JsonRpcError) {
                        JsonRpcError jre = (JsonRpcError) cause;
                        outputArea.setText("JSON-RPC error " + jre.getCode() + ": " + jre.getMessage());
                        rawArea.setText(jre.getRawResponse() == null ? "" : jre.getRawResponse());
                    } else {
                        outputArea.setText("summarize failed: " + cause.getMessage());
                    }
                    timingLbl.setText("timing: —");
                }
            }
        }.execute();
    }
}

