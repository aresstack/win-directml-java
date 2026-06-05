package com.aresstack.windirectml.inference.t5;

/**
 * Signals that the T5 runtime API shell was called before WARP execution exists.
 */
public final class T5UnsupportedRuntimeException extends UnsupportedOperationException {
    public T5UnsupportedRuntimeException(String message) {
        super(message);
    }
}
