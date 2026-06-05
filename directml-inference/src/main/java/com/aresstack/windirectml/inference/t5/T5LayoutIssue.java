package com.aresstack.windirectml.inference.t5;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Describes one diagnostic issue found while validating a T5 source layout.
 */
final class T5LayoutIssue {
    private final String kind;
    private final String message;

    T5LayoutIssue(String kind, String message) {
        this.kind = Objects.requireNonNull(kind, "kind");
        this.message = Objects.requireNonNull(message, "message");
    }

    String kind() {
        return kind;
    }

    String message() {
        return message;
    }

    Map<String, Object> toManifest() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("kind", kind);
        out.put("message", message);
        return out;
    }
}
