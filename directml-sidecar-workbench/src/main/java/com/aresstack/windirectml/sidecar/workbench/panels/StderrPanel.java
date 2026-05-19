package com.aresstack.windirectml.sidecar.workbench.panels;

import com.aresstack.windirectml.sidecar.workbench.WorkbenchModel;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public final class StderrPanel extends JPanel {

    private static final long serialVersionUID = 1L;

    private final WorkbenchModel model;
    private final JTextArea area = new JTextArea(20, 80);
    private int lastLen = 0;

    public StderrPanel(WorkbenchModel model) {
        this.model = model;
        setLayout(new BorderLayout(0, 6));
        setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

        area.setEditable(false);
        area.setLineWrap(false);

        JButton clearBtn = new JButton("Clear");
        clearBtn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                area.setText("");
                lastLen = 0;
            }
        });

        JScrollPane sp = new JScrollPane(area);
        sp.setBorder(BorderFactory.createTitledBorder("Sidecar stderr (logs / diagnostics)"));
        add(sp, BorderLayout.CENTER);
        add(clearBtn, BorderLayout.SOUTH);
    }

    public void refresh() {
        String s = model.getStderrSnapshot();
        if (s == null) return;
        // Only append delta to keep the UI responsive even on large logs.
        if (s.length() > lastLen) {
            String delta = s.substring(lastLen);
            area.append(delta);
            lastLen = s.length();
            area.setCaretPosition(area.getDocument().getLength());
        } else if (s.length() < lastLen) {
            // Buffer was rotated / trimmed in the client; restart.
            area.setText(s);
            lastLen = s.length();
        }
    }
}

