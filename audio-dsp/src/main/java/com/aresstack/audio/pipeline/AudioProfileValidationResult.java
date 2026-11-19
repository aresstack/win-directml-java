package com.aresstack.audio.pipeline;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** The Swing-free outcome of validating a profile: the collected issues plus convenient summaries. */
public final class AudioProfileValidationResult {

    private final List<AudioProfileValidationIssue> issues;

    public AudioProfileValidationResult(List<AudioProfileValidationIssue> issues) {
        this.issues = Collections.unmodifiableList(new ArrayList<AudioProfileValidationIssue>(
                issues == null ? new ArrayList<AudioProfileValidationIssue>() : issues));
    }

    public List<AudioProfileValidationIssue> getIssues() {
        return issues;
    }

    public boolean hasErrors() {
        return errorCount() > 0;
    }

    public boolean hasWarnings() {
        return warningCount() > 0;
    }

    public int errorCount() {
        return count(AudioValidationSeverity.ERROR);
    }

    public int warningCount() {
        return count(AudioValidationSeverity.WARNING);
    }

    /** @return all issues for one block id, in order (empty when the block has none). */
    public List<AudioProfileValidationIssue> issuesForBlock(String blockId) {
        List<AudioProfileValidationIssue> result = new ArrayList<AudioProfileValidationIssue>();
        for (AudioProfileValidationIssue issue : issues) {
            if (issue.getBlockId() != null && issue.getBlockId().equals(blockId)) {
                result.add(issue);
            }
        }
        return result;
    }

    /** @return the first error issue, or null when there are none. */
    public AudioProfileValidationIssue firstError() {
        for (AudioProfileValidationIssue issue : issues) {
            if (issue.getSeverity() == AudioValidationSeverity.ERROR) {
                return issue;
            }
        }
        return null;
    }

    private int count(AudioValidationSeverity severity) {
        int count = 0;
        for (AudioProfileValidationIssue issue : issues) {
            if (issue.getSeverity() == severity) {
                count++;
            }
        }
        return count;
    }
}
