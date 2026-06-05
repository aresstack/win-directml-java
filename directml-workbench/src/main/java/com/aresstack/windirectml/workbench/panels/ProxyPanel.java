package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.workbench.WorkbenchModel;
import com.aresstack.winproxy.ProxyConfiguration;
import com.aresstack.winproxy.ProxyDefaults;
import com.aresstack.winproxy.ProxyMode;
import com.aresstack.winproxy.ProxyResult;
import com.aresstack.winproxy.WindowsProxyResolver;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import javax.swing.SwingWorker;
import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.GridLayout;

/**
 * Proxy settings used by model downloads.
 */
public final class ProxyPanel extends JPanel {
    private final WorkbenchModel model;
    private final JComboBox<ProxyMode> mode;
    private final JTextField testUrl;
    private final JTextField pacUrlDiscoveryScript;
    private final JTextField pacUrl;
    private final JTextField manualHost;
    private final JTextField manualPort;
    private final JTextArea log;

    public ProxyPanel(WorkbenchModel model) {
        this.model = model;
        ProxyConfiguration cfg = model.getProxyConfiguration();
        setLayout(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        mode = new JComboBox<ProxyMode>(ProxyMode.values());
        mode.setSelectedItem(cfg.getMode());
        testUrl = new JTextField(cfg.getTestUrl());
        pacUrl = new JTextField(empty(cfg.getPacUrl()));
        manualHost = new JTextField(empty(cfg.getManualProxyHost()));
        manualPort = new JTextField(cfg.getManualProxyPort() > 0 ? String.valueOf(cfg.getManualProxyPort()) : "");

        JPanel form = new JPanel(new GridLayout(0, 2, 6, 6));
        form.setBorder(BorderFactory.createTitledBorder("Proxy for model downloads"));
        form.add(new JLabel("Mode"));
        form.add(mode);
        form.add(new JLabel("Test URL"));
        form.add(testUrl);
        form.add(new JLabel("Explicit PAC URL (optional)"));
        form.add(pacUrl);
        form.add(new JLabel("Manual host"));
        form.add(manualHost);
        form.add(new JLabel("Manual port"));
        form.add(manualPort);

        JPanel buttons = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JButton apply = new JButton("Apply to downloads");
        apply.addActionListener(e -> apply());
        JButton test = new JButton("Resolve test URL");
        test.addActionListener(e -> resolveTestUrl());
        JButton defaults = new JButton("Reset default");
        defaults.addActionListener(e -> resetDefault());
        buttons.add(apply);
        buttons.add(test);
        buttons.add(defaults);

        JPanel top = new JPanel(new BorderLayout(8, 8));
        top.add(form, BorderLayout.CENTER);
        top.add(buttons, BorderLayout.SOUTH);
        add(top, BorderLayout.NORTH);

        log = new JTextArea(12, 60);
        log.setEditable(false);
        add(new JScrollPane(log), BorderLayout.CENTER);
        append("Default is PAC_URL: Windows AutoConfigURL/WPAD -> PAC download -> FindProxyForURL.");
    }

    private void apply() {
        ProxyConfiguration cfg = buildConfiguration();
        model.setProxyConfiguration(cfg);
        append("Applied proxy mode for downloads: " + cfg.getMode());
    }

    private void resetDefault() {
        ProxyConfiguration cfg = ProxyConfiguration.defaults();
        model.setProxyConfiguration(cfg);
        mode.setSelectedItem(cfg.getMode());
        testUrl.setText(cfg.getTestUrl());
        pacUrl.setText("");
        manualHost.setText("");
        manualPort.setText("");
        append("Reset to win-proxy-java defaults: " + cfg.getMode());
    }

    private void resolveTestUrl() {
        ProxyConfiguration cfg = buildConfiguration();
        model.setProxyConfiguration(cfg);
        String url = cfg.getTestUrl();
        append("Resolving " + url + " ...");
        new SwingWorker<ProxyResult, Void>() {
            @Override
            protected ProxyResult doInBackground() {
                return new WindowsProxyResolver(cfg).resolve(url);
            }

            @Override
            protected void done() {
                try {
                    ProxyResult result = get();
                    append(result.isDirect() ? "Result: DIRECT" : "Result: " + result.getHost() + ":" + result.getPort());
                } catch (Exception ex) {
                    append("ERROR: " + ex.getMessage());
                }
            }
        }.execute();
    }

    private ProxyConfiguration buildConfiguration() {
        ProxyConfiguration.Builder builder = ProxyConfiguration.builder()
                .mode(selectedMode())
                .testUrl(nonBlank(testUrl.getText(), ProxyConfiguration.defaults().getTestUrl()));
        String pac = trimToNull(pacUrl.getText());
        if (pac != null) {
            builder.pacUrl(pac);
        }
        String host = trimToNull(manualHost.getText());
        if (host != null) {
            builder.manualProxyHost(host);
        }
        Integer port = parsePort(manualPort.getText());
        if (port != null) {
            builder.manualProxyPort(port.intValue());
        }
        return builder.build();
    }

    private ProxyMode selectedMode() {
        Object selected = mode.getSelectedItem();
        return selected instanceof ProxyMode ? (ProxyMode) selected : ProxyMode.PAC_URL;
    }

    private void append(String message) {
        SwingUtilities.invokeLater(() -> log.append(message + System.lineSeparator()));
    }

    private static String nonBlank(String value, String fallback) {
        String trimmed = trimToNull(value);
        return trimmed == null ? fallback : trimmed;
    }

    private static String trimToNull(String value) {
        if (value == null) {
            return null;
        }
        String trimmed = value.trim();
        return trimmed.isEmpty() ? null : trimmed;
    }

    private static String empty(String value) {
        return value == null ? "" : value;
    }

    private static Integer parsePort(String value) {
        String trimmed = trimToNull(value);
        if (trimmed == null) {
            return null;
        }
        try {
            return Integer.valueOf(trimmed);
        } catch (NumberFormatException ex) {
            return null;
        }
    }
}
