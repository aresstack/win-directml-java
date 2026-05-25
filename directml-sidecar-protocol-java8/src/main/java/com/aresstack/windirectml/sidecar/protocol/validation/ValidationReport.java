package com.aresstack.windirectml.sidecar.protocol.validation;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class ValidationReport {
    private final String label;
    private final File directory;
    private final List<ValidationFinding> findings;

    public ValidationReport(String label, File directory, List<ValidationFinding> findings) {
        this.label = label == null ? "model" : label;
        this.directory = directory;
        this.findings = findings == null ? Collections.<ValidationFinding>emptyList()
                : Collections.unmodifiableList(new ArrayList<ValidationFinding>(findings));
    }

    public String getLabel() { return label; }
    public File getDirectory() { return directory; }
    public List<ValidationFinding> getFindings() { return findings; }

    public boolean isOk() { return errorCount() == 0; }

    public int errorCount() {
        int count = 0;
        for (ValidationFinding finding : findings) {
            if (finding.isError()) count++;
        }
        return count;
    }

    public int warningCount() {
        int count = 0;
        for (ValidationFinding finding : findings) {
            if (finding.isWarning()) count++;
        }
        return count;
    }

    public String format() {
        StringBuilder out = new StringBuilder();
        out.append("== ").append(label).append(" ==");
        if (directory != null) out.append("  (").append(directory.getPath()).append(")");
        out.append('\n');
        for (ValidationFinding finding : findings) {
            out.append('[').append(finding.getSeverity()).append("] ")
                    .append(finding.getMessage()).append('\n');
        }
        out.append("Result: ").append(isOk() ? "OK" : "NOT OK");
        out.append(" (").append(errorCount()).append(" error(s), ")
                .append(warningCount()).append(" warning(s))");
        return out.toString();
    }

    public String toString() { return format(); }
}
