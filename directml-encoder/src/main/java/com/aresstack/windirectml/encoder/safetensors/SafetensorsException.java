package com.aresstack.windirectml.encoder.safetensors;

/**
 * Signalisiert defekte oder nicht unterstützte Safetensors-Dateien.
 */
public class SafetensorsException extends Exception {

    public SafetensorsException(String message) {
        super(message);
    }

    public SafetensorsException(String message, Throwable cause) {
        super(message, cause);
    }
}

