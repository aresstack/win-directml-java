package com.aresstack.audio.enhance;

/**
 * Availability of an optional speech-enhancement backend. A missing backend never breaks a profile — the
 * block stays editable and the editor shows the reason; the pure-Java core keeps working without it.
 */
public enum BackendAvailability {
    AVAILABLE("Available"),
    NOT_INSTALLED("Not installed"),
    UNSUPPORTED_PLATFORM("Unsupported platform"),
    MISSING_MODEL("Missing model"),
    INVALID_SAMPLE_RATE("Invalid sample rate");

    private final String displayName;

    BackendAvailability(String displayName) {
        this.displayName = displayName;
    }

    public String getDisplayName() {
        return displayName;
    }

    public boolean isRunnable() {
        return this == AVAILABLE;
    }
}
