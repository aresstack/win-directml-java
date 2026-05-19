package com.aresstack.windirectml.sidecar.workbench;

import com.aresstack.windirectml.sidecar.workbench.panels.ConfigPanel;
import com.aresstack.windirectml.sidecar.workbench.panels.EmbeddingsPanel;
import com.aresstack.windirectml.sidecar.workbench.panels.HealthPanel;
import com.aresstack.windirectml.sidecar.workbench.panels.InspectorPanel;
import com.aresstack.windirectml.sidecar.workbench.panels.IntegrationHelpPanel;
import com.aresstack.windirectml.sidecar.workbench.panels.StderrPanel;
import com.aresstack.windirectml.sidecar.workbench.panels.SummarizerPanel;

import javax.swing.BorderFactory;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import javax.swing.SwingConstants;
import javax.swing.Timer;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

/**
 * Top-level workbench window. Java 8 / Swing.
 *
 * <p>Holds the {@link WorkbenchModel} and dispatches it to all tab panels.
 */
public final class WorkbenchFrame extends JFrame {

    private static final long serialVersionUID = 1L;

    private final WorkbenchModel model = new WorkbenchModel();
    private final JLabel statusLabel = new JLabel(" status: not started ", SwingConstants.LEFT);

    private final ConfigPanel configPanel;
    private final HealthPanel healthPanel;
    private final EmbeddingsPanel embeddingsPanel;
    private final SummarizerPanel summarizerPanel;
    private final InspectorPanel inspectorPanel;
    private final StderrPanel stderrPanel;
    private final IntegrationHelpPanel integrationPanel;

    public WorkbenchFrame() {
        super("DirectML Sidecar Workbench");
        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        setSize(new Dimension(1100, 760));
        setLocationByPlatform(true);

        configPanel = new ConfigPanel(model);
        healthPanel = new HealthPanel(model);
        embeddingsPanel = new EmbeddingsPanel(model);
        summarizerPanel = new SummarizerPanel(model);
        inspectorPanel = new InspectorPanel(model);
        stderrPanel = new StderrPanel(model);
        integrationPanel = new IntegrationHelpPanel();

        JTabbedPane tabs = new JTabbedPane();
        tabs.addTab("Config & Control", configPanel);
        tabs.addTab("Health", healthPanel);
        tabs.addTab("Embeddings", embeddingsPanel);
        tabs.addTab("Summarize", summarizerPanel);
        tabs.addTab("JSON-RPC Inspector", inspectorPanel);
        tabs.addTab("stderr Log", stderrPanel);
        tabs.addTab("Integration Help", integrationPanel);

        JPanel statusBar = new JPanel(new BorderLayout());
        statusBar.setBorder(BorderFactory.createMatteBorder(1, 0, 0, 0, Color.LIGHT_GRAY));
        statusBar.add(statusLabel, BorderLayout.WEST);

        setLayout(new BorderLayout());
        add(tabs, BorderLayout.CENTER);
        add(statusBar, BorderLayout.SOUTH);

        // Wire shared refresh actions across panels.
        configPanel.setOnChange(new Runnable() {
            @Override
            public void run() {
                refreshAll();
            }
        });

        // Periodic UI refresh: status + inspector + stderr.
        Timer t = new Timer(500, new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                updateStatusBar();
                inspectorPanel.refresh();
                stderrPanel.refresh();
            }
        });
        t.start();
    }

    private void refreshAll() {
        updateStatusBar();
        healthPanel.refresh();
        inspectorPanel.refresh();
        stderrPanel.refresh();
    }

    private void updateStatusBar() {
        boolean running = model.isRunning();
        int exit = model.exitValue();
        String s = running
                ? " status: running"
                : " status: not running" + (exit > 0 ? " (last exit " + exit + ")" : "");
        statusLabel.setText(s);
        statusLabel.setForeground(running ? new Color(0, 110, 0) : Color.DARK_GRAY);
    }

    // Test hook.
    WorkbenchModel getModelForTesting() {
        return model;
    }
}

