package com.aresstack.windirectml.runtime;

/**
 * Geprüfte Ausnahme der DirectML-Runtime-Schicht.
 * <p>
 * Wickelt HRESULT-Fehler, COM-Fehler und Treiber-Probleme einheitlich ein.
 */
public class DirectMlRuntimeException extends Exception {

    public DirectMlRuntimeException(String message) {
        super(message);
    }

    public DirectMlRuntimeException(String message, Throwable cause) {
        super(message, cause);
    }
}

