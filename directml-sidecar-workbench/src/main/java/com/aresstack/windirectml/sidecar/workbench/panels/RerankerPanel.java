package com.aresstack.windirectml.sidecar.workbench.panels;

import com.aresstack.windirectml.config.InputLimits;
import com.aresstack.windirectml.sidecar.client.RerankResult;
import com.aresstack.windirectml.sidecar.workbench.WorkbenchModel;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSpinner;
import javax.swing.JTable;
import javax.swing.JTextArea;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingWorker;
import javax.swing.table.DefaultTableModel;

import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;

/**
 * Cross-encoder reranking tab. Lets the operator paste a query plus a
 * blank-line-separated list of candidate documents, runs the sidecar's
 * {@code rerank} method and renders the result table sorted by
 * descending score.
 * <p>
 * The candidate list intentionally uses the empty-line separator
 * convention (instead of one-per-line) so multi-line documents –
 * paragraphs, code snippets, FAQ entries – can be tested without
 * needing a dedicated editor widget per row.
 */
public final class RerankerPanel extends JPanel {

    private static final long serialVersionUID = 1L;

    private final WorkbenchModel model;
    private final JTextArea queryArea = new JTextArea(2, 60);
    private final JTextArea docsArea = new JTextArea(12, 60);
    private final JSpinner topNSpinner = new JSpinner(new SpinnerNumberModel(5, 0, 1000, 1));
    private final JLabel statusLbl = new JLabel("idle");
    private final DefaultTableModel resultModel = new DefaultTableModel(
            new Object[]{"rank", "score", "original index", "document"}, 0) {
        private static final long serialVersionUID = 1L;
        @Override public boolean isCellEditable(int row, int col) { return false; }
    };
    private final JTable resultTable = new JTable(resultModel);

    public RerankerPanel(WorkbenchModel model) {
        this.model = model;
        setLayout(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

        queryArea.setText("What is DirectML?");
        docsArea.setText(String.join("\n\n",
                "DirectML is a low-level Windows API for hardware-accelerated machine learning.",
                "BERT is a transformer encoder commonly used for sentence embeddings.",
                "Cross-encoder rerankers score (query, document) pairs jointly for relevance.",
                "Python is a popular programming language for data science."));

        JPanel north = new JPanel(new GridLayout(2, 1, 0, 4));
        north.add(titledScroll(queryArea, "Query"));
        north.add(titledScroll(docsArea, "Documents (separated by a blank line)"));

        JButton rerankBtn = new JButton("Rerank");
        rerankBtn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                runRerank();
            }
        });

        JPanel controls = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 0));
        controls.add(new JLabel("top N (0 = all):"));
        controls.add(topNSpinner);
        controls.add(rerankBtn);
        controls.add(statusLbl);

        JLabel scoreHint = new JLabel(
                "<html><i>Scores are raw model logits. Compare them only within the same query, "
                        + "not across models or queries.</i></html>");
        scoreHint.setBorder(BorderFactory.createEmptyBorder(4, 2, 4, 2));

        JPanel center = new JPanel(new BorderLayout(0, 6));
        center.add(controls, BorderLayout.NORTH);
        JScrollPane tableScroll = new JScrollPane(resultTable);
        tableScroll.setBorder(BorderFactory.createTitledBorder("Ranked results"));
        center.add(tableScroll, BorderLayout.CENTER);
        center.add(scoreHint, BorderLayout.SOUTH);

        add(north, BorderLayout.NORTH);
        add(center, BorderLayout.CENTER);
    }

    private static JScrollPane titledScroll(JTextArea ta, String title) {
        JScrollPane sp = new JScrollPane(ta);
        sp.setBorder(BorderFactory.createTitledBorder(title));
        ta.setLineWrap(true);
        ta.setWrapStyleWord(true);
        return sp;
    }

    private void runRerank() {
        if (!model.isRunning()) {
            statusLbl.setText("sidecar not running");
            return;
        }
        final String query = queryArea.getText().trim();
        if (query.isEmpty()) {
            statusLbl.setText("query must not be empty");
            return;
        }
        int maxTextLen = InputLimits.maxTextLength();
        if (query.length() > maxTextLen) {
            statusLbl.setText("query too long: " + query.length() + " chars (max " + maxTextLen + ")");
            return;
        }
        final List<String> docs = splitDocuments(docsArea.getText());
        if (docs.isEmpty()) {
            statusLbl.setText("no candidate documents");
            return;
        }
        int maxDocs = InputLimits.maxRerankDocuments();
        if (docs.size() > maxDocs) {
            statusLbl.setText("too many documents: " + docs.size() + " (max " + maxDocs + ")");
            return;
        }
        int maxDocLen = InputLimits.maxRerankDocumentLength();
        for (int i = 0; i < docs.size(); i++) {
            if (docs.get(i).length() > maxDocLen) {
                statusLbl.setText("document " + (i + 1) + " too long: "
                        + docs.get(i).length() + " chars (max " + maxDocLen + ")");
                return;
            }
        }
        final int topN = ((Number) topNSpinner.getValue()).intValue();

        statusLbl.setText("running…");
        resultModel.setRowCount(0);
        new SwingWorker<RerankResult, Void>() {
            @Override
            protected RerankResult doInBackground() throws Exception {
                return model.rerank(query, docs, topN);
            }

            @Override
            protected void done() {
                try {
                    RerankResult r = get();
                    List<RerankResult.Item> items = r.getItems();
                    for (int i = 0; i < items.size(); i++) {
                        RerankResult.Item it = items.get(i);
                        String doc = (it.getIndex() >= 0 && it.getIndex() < docs.size())
                                ? docs.get(it.getIndex())
                                : "<index " + it.getIndex() + " out of range>";
                        resultModel.addRow(new Object[]{
                                i + 1,
                                String.format("%.6f", it.getScore()),
                                it.getIndex(),
                                preview(doc)
                        });
                    }
                    statusLbl.setText("model: " + (r.getModel() == null ? "—" : r.getModel())
                            + "  |  " + items.size() + " results in " + r.getElapsedMillis() + " ms");
                } catch (Exception ex) {
                    Throwable cause = (ex.getCause() != null) ? ex.getCause() : ex;
                    statusLbl.setText("rerank failed: " + cause.getMessage());
                }
            }
        }.execute();
    }

    /**
     * Split the documents text-area on one-or-more blank lines.
     * Visible-for-test so the parsing rule can be exercised without a
     * Swing dispatch loop.
     */
    static List<String> splitDocuments(String text) {
        List<String> out = new ArrayList<String>();
        if (text == null) return out;
        String[] chunks = text.split("\\r?\\n\\s*\\r?\\n");
        for (String c : chunks) {
            String trimmed = c.trim();
            if (!trimmed.isEmpty()) out.add(trimmed);
        }
        return out;
    }

    private static String preview(String s) {
        if (s == null) return "";
        s = s.replace('\n', ' ').replace('\r', ' ');
        return s.length() <= 120 ? s : s.substring(0, 117) + "…";
    }
}

