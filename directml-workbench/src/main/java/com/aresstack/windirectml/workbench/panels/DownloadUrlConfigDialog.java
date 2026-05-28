package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.workbench.download.ModelDownloadManifest;
import com.aresstack.windirectml.workbench.download.ModelFileDescriptor;

import javax.swing.*;
import java.awt.*;
import java.awt.datatransfer.StringSelection;
import java.util.ArrayList;
import java.util.List;

/**
 * Dialog that shows all downloadable files for a model with editable URL fields.
 *
 * <p>Each file row contains:
 * <ul>
 *   <li>Label showing the local filename and required/optional marker</li>
 *   <li>Editable text field containing the download URL</li>
 *   <li>Square icon-only copy button to copy that single URL</li>
 * </ul>
 *
 * <p>OK accepts edits; Cancel discards them.
 */
public final class DownloadUrlConfigDialog extends JDialog {

    private final List<JTextField> urlFields = new ArrayList<>();
    private boolean accepted = false;

    public DownloadUrlConfigDialog(Window owner, ModelDownloadManifest manifest) {
        super(owner, "Configure download URLs – " + manifest.modelId(), ModalityType.APPLICATION_MODAL);
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);
        buildUI(manifest);
        pack();
        setMinimumSize(new Dimension(700, 300));
        setLocationRelativeTo(owner);
    }

    private void buildUI(ModelDownloadManifest manifest) {
        var contentPanel = new JPanel(new BorderLayout(8, 8));
        contentPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        // File rows
        var filesPanel = new JPanel(new GridBagLayout());
        var gbc = new GridBagConstraints();
        gbc.insets = new Insets(2, 2, 2, 2);
        gbc.fill = GridBagConstraints.HORIZONTAL;

        int row = 0;
        for (ModelFileDescriptor desc : manifest.files()) {
            // Label
            String labelText = desc.displayName()
                    + (desc.required() ? " (required)" : " (optional)");
            gbc.gridx = 0;
            gbc.gridy = row;
            gbc.weightx = 0;
            gbc.fill = GridBagConstraints.NONE;
            gbc.anchor = GridBagConstraints.WEST;
            filesPanel.add(new JLabel(labelText), gbc);

            // URL text field
            var urlField = new JTextField(desc.currentUrl(), 50);
            urlFields.add(urlField);
            gbc.gridx = 1;
            gbc.weightx = 1;
            gbc.fill = GridBagConstraints.HORIZONTAL;
            filesPanel.add(urlField, gbc);

            // Copy button
            var copyBtn = new JButton("\uD83D\uDCCB");
            copyBtn.setToolTipText("Copy address");
            copyBtn.setMargin(new Insets(0, 0, 0, 0));
            Dimension sq = new Dimension(28, 28);
            copyBtn.setPreferredSize(sq);
            copyBtn.setMinimumSize(sq);
            copyBtn.setMaximumSize(sq);
            final int fieldIndex = row;
            copyBtn.addActionListener(e -> {
                String url = urlFields.get(fieldIndex).getText();
                var selection = new StringSelection(url);
                Toolkit.getDefaultToolkit().getSystemClipboard().setContents(selection, null);
            });
            gbc.gridx = 2;
            gbc.weightx = 0;
            gbc.fill = GridBagConstraints.NONE;
            filesPanel.add(copyBtn, gbc);

            row++;
        }

        contentPanel.add(new JScrollPane(filesPanel), BorderLayout.CENTER);

        // OK / Cancel buttons
        var buttonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        var okBtn = new JButton("OK");
        okBtn.addActionListener(e -> {
            accepted = true;
            dispose();
        });
        var cancelBtn = new JButton("Cancel");
        cancelBtn.addActionListener(e -> dispose());
        buttonPanel.add(okBtn);
        buttonPanel.add(cancelBtn);
        contentPanel.add(buttonPanel, BorderLayout.SOUTH);

        setContentPane(contentPanel);
    }

    /**
     * Returns true if the user pressed OK.
     */
    public boolean isAccepted() {
        return accepted;
    }

    /**
     * Returns the list of URLs as edited by the user.
     * Only valid after the dialog is disposed and {@link #isAccepted()} is true.
     */
    public List<String> getEditedUrls() {
        var urls = new ArrayList<String>();
        for (var field : urlFields) {
            urls.add(field.getText());
        }
        return List.copyOf(urls);
    }
}
