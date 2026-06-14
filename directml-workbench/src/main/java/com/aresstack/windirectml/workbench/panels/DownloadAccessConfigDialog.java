package com.aresstack.windirectml.workbench.panels;

import com.aresstack.windirectml.workbench.download.DownloadAccessSettings;
import com.aresstack.windirectml.workbench.download.DownloadUrlOpener;
import com.aresstack.windirectml.workbench.download.ModelDownloadManifest;
import com.aresstack.windirectml.workbench.download.ModelFileDescriptor;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JPasswordField;
import javax.swing.JScrollPane;
import javax.swing.JTextField;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.HeadlessException;
import java.awt.Insets;
import java.awt.Toolkit;
import java.awt.Window;
import java.awt.datatransfer.StringSelection;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Dialog that edits download URLs and a Hugging Face token for gated model repositories.
 */
public final class DownloadAccessConfigDialog extends JDialog {

    private static final Logger LOG = Logger.getLogger(DownloadAccessConfigDialog.class.getName());
    private static final Insets ICON_BUTTON_MARGIN = new Insets(0, 0, 0, 0);
    private static final Dimension ICON_BUTTON_SIZE = new Dimension(28, 28);

    private final List<JTextField> urlFields = new ArrayList<JTextField>();
    private final JPasswordField huggingFaceTokenField;
    private boolean accepted = false;

    public DownloadAccessConfigDialog(Window owner, ModelDownloadManifest manifest,
                                      DownloadAccessSettings accessSettings) {
        super(owner, "Configure gated download – " + manifest.modelId(), ModalityType.APPLICATION_MODAL);
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);
        DownloadAccessSettings effectiveSettings = accessSettings == null
                ? DownloadAccessSettings.empty() : accessSettings;
        this.huggingFaceTokenField = new JPasswordField(effectiveSettings.huggingFaceToken(), 48);
        buildUserInterface(manifest);
        pack();
        setMinimumSize(new Dimension(760, 360));
        setLocationRelativeTo(owner);
    }

    private void buildUserInterface(ModelDownloadManifest manifest) {
        JPanel contentPanel = new JPanel(new BorderLayout(8, 8));
        contentPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        JPanel tokenPanel = new JPanel(new GridBagLayout());
        tokenPanel.setBorder(BorderFactory.createTitledBorder("Hugging Face access"));
        GridBagConstraints tokenConstraints = new GridBagConstraints();
        tokenConstraints.insets = new Insets(2, 2, 2, 2);
        tokenConstraints.gridx = 0;
        tokenConstraints.gridy = 0;
        tokenConstraints.anchor = GridBagConstraints.WEST;
        tokenPanel.add(new JLabel("HF token:"), tokenConstraints);

        tokenConstraints.gridx = 1;
        tokenConstraints.weightx = 1.0d;
        tokenConstraints.fill = GridBagConstraints.HORIZONTAL;
        huggingFaceTokenField.setToolTipText("Paste a Hugging Face read token for gated repositories. Leave blank to clear the stored token.");
        tokenPanel.add(huggingFaceTokenField, tokenConstraints);

        tokenConstraints.gridx = 1;
        tokenConstraints.gridy = 1;
        tokenConstraints.weightx = 1.0d;
        tokenConstraints.fill = GridBagConstraints.HORIZONTAL;
        tokenPanel.add(new JLabel("Accept the model terms in the browser first. The token is stored locally and never appended to URLs."),
                tokenConstraints);
        contentPanel.add(tokenPanel, BorderLayout.NORTH);

        JPanel filesPanel = new JPanel(new GridBagLayout());
        filesPanel.setBorder(BorderFactory.createTitledBorder("Effective download URLs"));
        addFileRows(filesPanel, manifest);
        contentPanel.add(new JScrollPane(filesPanel), BorderLayout.CENTER);

        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        JButton okButton = new JButton("OK");
        okButton.addActionListener(e -> accept());
        JButton cancelButton = new JButton("Cancel");
        cancelButton.addActionListener(e -> dispose());
        buttonPanel.add(okButton);
        buttonPanel.add(cancelButton);
        contentPanel.add(buttonPanel, BorderLayout.SOUTH);

        setContentPane(contentPanel);
    }

    private void addFileRows(JPanel filesPanel, ModelDownloadManifest manifest) {
        GridBagConstraints constraints = new GridBagConstraints();
        constraints.insets = new Insets(2, 2, 2, 2);
        constraints.fill = GridBagConstraints.HORIZONTAL;

        int row = 0;
        for (ModelFileDescriptor descriptor : manifest.files()) {
            String labelText = descriptor.displayName()
                    + (descriptor.required() ? " (required)" : " (optional)");
            constraints.gridx = 0;
            constraints.gridy = row;
            constraints.weightx = 0.0d;
            constraints.fill = GridBagConstraints.NONE;
            constraints.anchor = GridBagConstraints.WEST;
            filesPanel.add(new JLabel(labelText), constraints);

            JTextField urlField = new JTextField(descriptor.currentUrl(), 50);
            urlFields.add(urlField);
            constraints.gridx = 1;
            constraints.weightx = 1.0d;
            constraints.fill = GridBagConstraints.HORIZONTAL;
            filesPanel.add(urlField, constraints);

            final int fieldIndex = row;
            JButton copyButton = createIconButton("\uD83D\uDCCB", "Copy address",
                    "Copy URL for " + descriptor.localFilename());
            copyButton.addActionListener(e -> copyUrlToClipboard(fieldIndex));
            constraints.gridx = 2;
            constraints.weightx = 0.0d;
            constraints.fill = GridBagConstraints.NONE;
            filesPanel.add(copyButton, constraints);

            JButton browserButton = createIconButton("\uD83C\uDF10", "Open address in default browser",
                    "Open URL for " + descriptor.localFilename());
            browserButton.addActionListener(e -> DownloadUrlOpener.openInBrowser(urlFields.get(fieldIndex).getText(), this));
            constraints.gridx = 3;
            filesPanel.add(browserButton, constraints);

            row++;
        }
    }

    private JButton createIconButton(String text, String tooltip, String accessibleName) {
        JButton button = new JButton(text);
        button.setToolTipText(tooltip);
        button.setMargin(ICON_BUTTON_MARGIN);
        button.setIconTextGap(0);
        button.setPreferredSize(ICON_BUTTON_SIZE);
        button.setMinimumSize(ICON_BUTTON_SIZE);
        button.setMaximumSize(ICON_BUTTON_SIZE);
        button.getAccessibleContext().setAccessibleName(accessibleName);
        button.setFocusable(false);
        return button;
    }

    private void copyUrlToClipboard(int fieldIndex) {
        String url = urlFields.get(fieldIndex).getText();
        StringSelection selection = new StringSelection(url);
        try {
            Toolkit.getDefaultToolkit().getSystemClipboard().setContents(selection, null);
        } catch (HeadlessException | IllegalStateException | SecurityException ex) {
            LOG.log(Level.WARNING, "Could not copy URL to clipboard", ex);
            JOptionPane.showMessageDialog(this,
                    "Could not copy URL to clipboard: " + ex.getMessage(),
                    "Clipboard unavailable",
                    JOptionPane.WARNING_MESSAGE);
        }
    }

    private void accept() {
        accepted = true;
        dispose();
    }

    public boolean isAccepted() {
        return accepted;
    }

    public List<String> getEditedUrls() {
        ArrayList<String> urls = new ArrayList<String>();
        for (JTextField field : urlFields) {
            urls.add(field.getText().trim());
        }
        return List.copyOf(urls);
    }

    public DownloadAccessSettings getAccessSettings() {
        return new DownloadAccessSettings(new String(huggingFaceTokenField.getPassword()));
    }
}
