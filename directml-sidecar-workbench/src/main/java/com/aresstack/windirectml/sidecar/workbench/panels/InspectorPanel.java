package com.aresstack.windirectml.sidecar.workbench.panels;

import com.aresstack.windirectml.sidecar.workbench.WorkbenchModel;

import javax.swing.BorderFactory;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

import java.awt.GridLayout;

public final class InspectorPanel extends JPanel {

    private static final long serialVersionUID = 1L;

    private final WorkbenchModel model;
    private final JTextArea reqArea = new JTextArea(8, 60);
    private final JTextArea respArea = new JTextArea(8, 60);

    public InspectorPanel(WorkbenchModel model) {
        this.model = model;
        setLayout(new GridLayout(2, 1, 0, 6));
        setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

        reqArea.setEditable(false);
        respArea.setEditable(false);
        reqArea.setLineWrap(true);
        respArea.setLineWrap(true);

        JScrollPane sp1 = new JScrollPane(reqArea);
        sp1.setBorder(BorderFactory.createTitledBorder("Last raw request"));
        JScrollPane sp2 = new JScrollPane(respArea);
        sp2.setBorder(BorderFactory.createTitledBorder("Last raw response"));
        add(sp1);
        add(sp2);
    }

    public void refresh() {
        String req = model.getLastRawRequest();
        String resp = model.getLastRawResponse();
        if (req != null && !req.equals(reqArea.getText())) {
            reqArea.setText(req);
            reqArea.setCaretPosition(0);
        }
        if (resp != null && !resp.equals(respArea.getText())) {
            respArea.setText(resp);
            respArea.setCaretPosition(0);
        }
    }
}

