package com.aresstack.windirectml.sidecar.workbench.panels;

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

public final class HealthPanel extends JPanel {

    private static final long serialVersionUID = 1L;

    private final WorkbenchModel model;
    private final JLabel sidecarRunning   = new JLabel("—");
    private final JLabel embeddingReady   = new JLabel("—");
    private final JLabel embeddingBackend = new JLabel("—");
    private final JLabel modelLoaded      = new JLabel("—");
    private final JLabel mode             = new JLabel("—");
    private final JLabel lastError        = new JLabel("—");
    private final JTextArea rawArea       = new JTextArea(8, 60);

    public HealthPanel(WorkbenchModel model) {
        this.model = model;
        setLayout(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

        JPanel grid = new JPanel(new GridLayout(0, 2, 6, 4));
        grid.setBorder(BorderFactory.createTitledBorder("Status"));
        grid.add(new JLabel("sidecar running:"));   grid.add(sidecarRunning);
        grid.add(new JLabel("embeddingReady:"));    grid.add(embeddingReady);
        grid.add(new JLabel("embeddingBackend:"));  grid.add(embeddingBackend);
        grid.add(new JLabel("modelLoaded:"));       grid.add(modelLoaded);
        grid.add(new JLabel("mode:"));              grid.add(mode);
        grid.add(new JLabel("lastError:"));         grid.add(lastError);

        JButton refreshBtn = new JButton("Refresh");
        refreshBtn.addActionListener(new ActionListener() {
            @Override public void actionPerformed(ActionEvent e) { refresh(); }
        });

        JPanel top = new JPanel(new BorderLayout());
        top.add(grid, BorderLayout.CENTER);
        top.add(refreshBtn, BorderLayout.SOUTH);

        rawArea.setEditable(false);
        rawArea.setLineWrap(true);
        rawArea.setWrapStyleWord(false);

        add(top, BorderLayout.NORTH);
        JScrollPane sp = new JScrollPane(rawArea);
        sp.setBorder(BorderFactory.createTitledBorder("Last raw health JSON"));
        add(sp, BorderLayout.CENTER);
    }

    public void refresh() {
        sidecarRunning.setText(Boolean.toString(model.isRunning()));
        if (!model.isRunning()) {
            embeddingReady.setText("—");
            embeddingBackend.setText("—");
            modelLoaded.setText("—");
            mode.setText("—");
            lastError.setText("—");
            return;
        }
        new SwingWorker<HealthResult, Void>() {
            @Override protected HealthResult doInBackground() throws Exception {
                return model.health();
            }
            @Override protected void done() {
                try {
                    HealthResult h = get();
                    embeddingReady.setText(Boolean.toString(h.isEmbeddingReady()));
                    embeddingBackend.setText(safe(h.getEmbeddingBackend()));
                    modelLoaded.setText(Boolean.toString(h.isModelLoaded()));
                    mode.setText(safe(h.getMode()));
                    lastError.setText(safe(h.getLastError()));
                    rawArea.setText(h.getRaw() == null ? "" : h.getRaw());
                } catch (Exception ex) {
                    Throwable cause = (ex.getCause() != null) ? ex.getCause() : ex;
                    lastError.setText("health() failed: " + cause.getMessage());
                }
            }
        }.execute();
    }

    private static String safe(String s) { return s == null ? "—" : s; }
}

