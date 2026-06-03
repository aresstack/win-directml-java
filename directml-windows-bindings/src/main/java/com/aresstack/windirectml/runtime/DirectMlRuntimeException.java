package com.aresstack.windirectml.runtime;

/**
 * Ungeprüfte (unchecked) Ausnahme der DirectML-Runtime-Schicht.
 * <p>
 * Wickelt HRESULT-Fehler, COM-Fehler und Treiber-Probleme einheitlich ein.
 * Erweitert {@link RuntimeException}, damit Methoden sie werfen
 * können, ohne sie in der Signatur deklarieren zu müssen.
 */
public class DirectMlRuntimeException extends RuntimeException {

    public DirectMlRuntimeException(String message) {
        super(message);
    }

    public DirectMlRuntimeException(String message, Throwable cause) {
        super(message, cause);
    }
}

