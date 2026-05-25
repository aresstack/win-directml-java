package com.aresstack.windirectml.sidecar.workbench.panels;

import com.aresstack.windirectml.config.InputLimits;
import com.aresstack.windirectml.sidecar.client.BatchEmbeddingResult;
import com.aresstack.windirectml.sidecar.client.EmbeddingResult;
import com.aresstack.windirectml.sidecar.workbench.WorkbenchModel;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.SwingWorker;

import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;

/**
 * Workbench panel that drives the {@code embedBatch} sidecar method.
 *
 * <p>The user pastes one text per line into the input area, picks a
 * shared prefix (none / query / passage), and hits "Embed Batch". The
 * panel renders the resulting matrix shape, per-vector previews, and
 * pairwise cosine similarities for the first few rows.
 *
 * <p>This is intentionally a thin UI on top of
 * {@link WorkbenchModel#embedBatch(java.util.List, boolean, String)} –
 * the heavy lifting (bucket-batched DirectML forward) lives in the
 * sidecar.
 */
public final class BatchEmbeddingsPanel extends JPanel {

    private static final long serialVersionUID = 1L;

    private static final String PREFIX_NONE = "none";
    private static final String PREFIX_QUERY = "query: ";
    private static final String PREFIX_PASSAGE = "passage: ";
    private static final String[] PREFIX_CHOICES = { PREFIX_NONE, PREFIX_QUERY, PREFIX_PASSAGE };

    private final WorkbenchModel model;

    private final JTextArea input = new JTextArea(10, 60);
    private final JComboBox<String> prefixBox = new JComboBox<String>(PREFIX_CHOICES);
    private final JLabel countLbl   = new JLabel("count: —");
    private final JLabel dimLbl     = new JLabel("dimension: —");
    private final JLabel modelLbl   = new JLabel("model: —");
    private final JLabel timingLbl  = new JLabel("timing: —");
    private final JLabel normLbl    = new JLabel("normalized: —");
    private final JTextArea output  = new JTextArea(14, 60);

    public BatchEmbeddingsPanel(WorkbenchModel model) {
        this.model = model;
        setLayout(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

        input.setText(String.join("\n",
                "DirectML is a low-level Windows API for ML.",
                "Cross-encoders score (query, document) pairs.",
                "Python is a programming language.",
                "BERT is a transformer-based encoder."));
        // Default for E5-style ingestion: passage-prefix, normalize.
        prefixBox.setSelectedItem(PREFIX_PASSAGE);

        JPanel header = new JPanel(new BorderLayout(4, 2));
        header.add(titledScroll(input, "Texts (one per line)"), BorderLayout.CENTER);
        header.add(prefixRow("Prefix", prefixBox), BorderLayout.SOUTH);

        JButton embedBtn = new JButton("Embed Batch");
        embedBtn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                runEmbedBatch();
            }
        });

        JPanel buttons = new JPanel();
        buttons.add(embedBtn);

        JPanel readouts = new JPanel(new GridLayout(3, 2, 6, 2));
        readouts.add(countLbl);
        readouts.add(modelLbl);
        readouts.add(dimLbl);
        readouts.add(timingLbl);
        readouts.add(normLbl);
        readouts.add(new JLabel());

        output.setEditable(false);
        output.setLineWrap(true);

        JPanel center = new JPanel(new BorderLayout(0, 6));
        center.add(buttons, BorderLayout.NORTH);
        center.add(readouts, BorderLayout.CENTER);
        center.add(titledScroll(output, "Result preview"), BorderLayout.SOUTH);

        add(header, BorderLayout.NORTH);
        add(center, BorderLayout.CENTER);
    }

    private static JScrollPane titledScroll(JTextArea ta, String title) {
        JScrollPane sp = new JScrollPane(ta);
        sp.setBorder(BorderFactory.createTitledBorder(title));
        return sp;
    }

    private static JPanel prefixRow(String label, JComboBox<String> box) {
        JPanel p = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 0));
        p.add(new JLabel(label + ":"));
        p.add(box);
        return p;
    }

    static String selectedPrefix(JComboBox<String> box) {
        String s = (String) box.getSelectedItem();
        if (s == null || PREFIX_NONE.equals(s)) return null;
        return s;
    }

    /** Visible-for-testing parser: split textarea into non-blank lines. */
    static List<String> parseTexts(String blob) {
        List<String> out = new ArrayList<String>();
        if (blob == null) return out;
        String[] lines = blob.split("\\r?\\n");
        for (String l : lines) {
            String trimmed = l.trim();
            if (trimmed.length() > 0) out.add(trimmed);
        }
        return out;
    }

    private void runEmbedBatch() {
        if (!model.isRunning()) {
            output.setText("sidecar not running");
            return;
        }
        final List<String> texts = parseTexts(input.getText());
        if (texts.isEmpty()) {
            output.setText("no non-blank lines to embed");
            return;
        }
        int maxBatch = InputLimits.maxEmbedBatchSize();
        if (texts.size() > maxBatch) {
            output.setText("too many texts: " + texts.size() + " (max " + maxBatch + ")");
            return;
        }
        int maxLen = InputLimits.maxTextLength();
        for (int i = 0; i < texts.size(); i++) {
            if (texts.get(i).length() > maxLen) {
                output.setText("text line " + (i + 1) + " too long: "
                        + texts.get(i).length() + " chars (max " + maxLen + ")");
                return;
            }
        }
        final String prefix = selectedPrefix(prefixBox);

        new SwingWorker<BatchEmbeddingResult, Void>() {
            @Override
            protected BatchEmbeddingResult doInBackground() throws Exception {
                return model.embedBatch(texts, true, prefix);
            }

            @Override
            protected void done() {
                try {
                    BatchEmbeddingResult r = get();
                    countLbl.setText("count: " + r.getCount());
                    dimLbl.setText("dimension: " + r.getDimension());
                    modelLbl.setText("model: " + safe(r.getModel()));
                    timingLbl.setText("timing: " + r.getElapsedMillis() + " ms");
                    normLbl.setText("normalized: " + r.isNormalized());

                    StringBuilder sb = new StringBuilder();
                    int preview = Math.min(6, r.getVectors().size());
                    int dims = Math.min(6, r.getDimension());
                    for (int i = 0; i < preview; i++) {
                        float[] v = r.getVectors().get(i);
                        sb.append("v[").append(i).append("][0..")
                                .append(dims - 1).append("] = [");
                        for (int j = 0; j < Math.min(dims, v.length); j++) {
                            if (j > 0) sb.append(", ");
                            sb.append(String.format("%.4f", v[j]));
                        }
                        sb.append("]\n");
                    }
                    if (r.getVectors().size() >= 2) {
                        sb.append("\ncosine(v0, v1) = ");
                        sb.append(String.format("%.6f",
                                EmbeddingResult.cosine(r.getVectors().get(0), r.getVectors().get(1))));
                    }
                    if (r.getVectors().size() >= 3) {
                        sb.append("\ncosine(v0, v2) = ");
                        sb.append(String.format("%.6f",
                                EmbeddingResult.cosine(r.getVectors().get(0), r.getVectors().get(2))));
                    }
                    output.setText(sb.toString());
                } catch (Exception ex) {
                    Throwable cause = (ex.getCause() != null) ? ex.getCause() : ex;
                    output.setText("embedBatch failed: " + cause.getMessage());
                }
            }
        }.execute();
    }

    private static String safe(String s) {
        return s == null ? "—" : s;
    }
}

