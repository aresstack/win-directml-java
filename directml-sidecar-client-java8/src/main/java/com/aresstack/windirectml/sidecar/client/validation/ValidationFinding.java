package com.aresstack.windirectml.sidecar.client.validation;

public final class ValidationFinding {
    public enum Severity { OK, WARN, ERROR }

    private final Severity severity;
    private final String message;

    public ValidationFinding(Severity severity, String message) {
        if (severity == null) throw new IllegalArgumentException("severity must not be null");
        if (message == null) throw new IllegalArgumentException("message must not be null");
        this.severity = severity;
        this.message = message;
    }

    public Severity getSeverity() { return severity; }
    public String getMessage() { return message; }
    public boolean isError() { return severity == Severity.ERROR; }
    public boolean isWarning() { return severity == Severity.WARN; }

    public static ValidationFinding ok(String message) { return new ValidationFinding(Severity.OK, message); }
    public static ValidationFinding warn(String message) { return new ValidationFinding(Severity.WARN, message); }
    public static ValidationFinding error(String message) { return new ValidationFinding(Severity.ERROR, message); }

    public String toString() { return severity + ": " + message; }
}
