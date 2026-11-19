package com.aresstack.audio.pipeline;

import com.aresstack.audio.profile.AudioBlockType;

/**
 * One validation finding, tied to a specific block and (when applicable) a specific parameter. Swing-free
 * so it can be produced by the validator and consumed by the editor without any UI coupling.
 */
public final class AudioProfileValidationIssue {

    private final AudioValidationSeverity severity;
    private final String blockId;
    private final AudioBlockType blockType;
    private final String parameterKey;
    private final String message;

    public AudioProfileValidationIssue(AudioValidationSeverity severity, String blockId,
                                       AudioBlockType blockType, String parameterKey, String message) {
        this.severity = severity;
        this.blockId = blockId;
        this.blockType = blockType;
        this.parameterKey = parameterKey;
        this.message = message;
    }

    public AudioValidationSeverity getSeverity() {
        return severity;
    }

    public String getBlockId() {
        return blockId;
    }

    public AudioBlockType getBlockType() {
        return blockType;
    }

    /** @return the affected parameter key, or null when the issue is not tied to a single parameter. */
    public String getParameterKey() {
        return parameterKey;
    }

    public String getMessage() {
        return message;
    }
}
