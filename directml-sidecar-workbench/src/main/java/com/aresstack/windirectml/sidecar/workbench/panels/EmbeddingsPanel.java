package com.aresstack.windirectml.sidecar.workbench.panels;

import com.aresstack.windirectml.sidecar.client.EmbeddingResult;
import com.aresstack.windirectml.sidecar.client.HealthResult;
import com.aresstack.windirectml.sidecar.workbench.WorkbenchModel;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.SwingWorker;

import java.awt.BorderLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public final class EmbeddingsPanel extends JPanel {

    private static final long serialVersionUID = 1L;

    private final WorkbenchModel model;

    private final JTextArea textA = new JTextArea(4, 60);
    private final JTextArea textB = new JTextArea(4, 60);
    private final JLabel dimLbl = new JLabel("dimension: —");
    private final JLabel backendLbl = new JLabel("backend: —");
    private final JLabel modelLbl = new JLabel("model: —");
    private final JLabel timingLbl = new JLabel("timing: —");
    private final JLabel cosineLbl = new JLabel("cos(A,B): —");
    private final JTextArea preview = new JTextArea(8, 60);

    private float[] vecA;
    private float[] vecB;

    public EmbeddingsPanel(WorkbenchModel model) {
        this.model = model;
        setLayout(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

        textA.setText("A cat sits on the mat.");
        textB.setText("A feline rests on a rug.");

        JPanel north = new JPanel(new GridLayout(2, 1, 0, 4));
        north.add(titledScroll(textA, "Text A"));
        north.add(titledScroll(textB, "Text B"));

        JButton embedA = new JButton("Embed A");
        JButton embedB = new JButton("Embed B");
        JButton cosineBtn = new JButton("Cosine Similarity");

        embedA.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                embed(textA.getText(), true);
            }
        });
        embedB.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                embed(textB.getText(), false);
            }
        });
        cosineBtn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (vecA == null || vecB == null) {
                    cosineLbl.setText("cos(A,B): — (embed both first)");
                } else {
                    double cs = EmbeddingResult.cosine(vecA, vecB);
                    cosineLbl.setText(String.format("cos(A,B): %.6f", cs));
                }
            }
        });

        JPanel buttons = new JPanel();
        buttons.add(embedA);
        buttons.add(embedB);
        buttons.add(cosineBtn);

        JPanel readouts = new JPanel(new GridLayout(3, 2, 6, 2));
        readouts.add(backendLbl);
        readouts.add(modelLbl);
        readouts.add(dimLbl);
        readouts.add(timingLbl);
        readouts.add(cosineLbl);
        readouts.add(new JLabel());

        preview.setEditable(false);
        preview.setLineWrap(true);

        JPanel center = new JPanel(new BorderLayout(0, 6));
        center.add(buttons, BorderLayout.NORTH);
        center.add(readouts, BorderLayout.CENTER);
        center.add(titledScroll(preview, "First N vector values"), BorderLayout.SOUTH);

        add(north, BorderLayout.NORTH);
        add(center, BorderLayout.CENTER);
    }

    private static JScrollPane titledScroll(JTextArea ta, String title) {
        JScrollPane sp = new JScrollPane(ta);
        sp.setBorder(BorderFactory.createTitledBorder(title));
        return sp;
    }

    private void embed(final String text, final boolean isA) {
        if (!model.isRunning()) {
            preview.setText("sidecar not running");
            return;
        }
        new SwingWorker<EmbeddingResult, Void>() {
            @Override
            protected EmbeddingResult doInBackground() throws Exception {
                return model.embed(text);
            }

            @Override
            protected void done() {
                try {
                    EmbeddingResult r = get();
                    if (isA) vecA = r.getVector();
                    else vecB = r.getVector();
                    dimLbl.setText("dimension: " + r.getDimension());
                    modelLbl.setText("model: " + safe(r.getModel()));
                    // Backend mode (cpu/directml/auto/...) is reported by
                    // health, not by embed. Refresh on demand so the user
                    // always sees which mode actually computed the vector.
                    try {
                        HealthResult h = model.health();
                        backendLbl.setText("backend: " + safe(h.getEmbeddingBackend())
                                + (h.isEmbeddingReady() ? "" : " (not ready)"));
                    } catch (Exception ignore) {
                        backendLbl.setText("backend: —");
                    }
                    timingLbl.setText("timing: " + r.getElapsedMillis() + " ms");
                    StringBuilder sb = new StringBuilder();
                    float[] v = r.getVector();
                    int n = Math.min(8, v.length);
                    sb.append(isA ? "A[0.." : "B[0..").append(n - 1).append("] = [");
                    for (int i = 0; i < n; i++) {
                        if (i > 0) sb.append(", ");
                        sb.append(String.format("%.4f", v[i]));
                    }
                    sb.append("]");
                    preview.setText(sb.toString());
                } catch (Exception ex) {
                    Throwable cause = (ex.getCause() != null) ? ex.getCause() : ex;
                    preview.setText("embed failed: " + cause.getMessage());
                }
            }
        }.execute();
    }

    private static String safe(String s) {
        return s == null ? "—" : s;
    }
}

