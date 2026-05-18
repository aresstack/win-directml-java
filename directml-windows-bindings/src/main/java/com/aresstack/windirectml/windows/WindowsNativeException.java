package com.aresstack.windirectml.windows;

/**
 * Exception thrown when a Windows native call fails.
 * <p>
 * Carries the raw {@code HRESULT} so callers can decide how to react.
 * {@link #getHresultDescription()} provides a human-readable name
 * (e.g. "E_INVALIDARG (0x80070057)") when the code is known.
 */
public class WindowsNativeException extends Exception {

    private final int hresult;

    public WindowsNativeException(String message) {
        super(message);
        this.hresult = 0;
    }

    public WindowsNativeException(String message, int hresult) {
        super(message);
        this.hresult = hresult;
    }

    public WindowsNativeException(String message, Throwable cause) {
        super(message, cause);
        this.hresult = 0;
    }

    /** The raw HRESULT returned by the failed Windows API call, or 0 if not applicable. */
    public int getHresult() {
        return hresult;
    }

    /**
     * Human-readable HRESULT description including symbolic name if known.
     * Returns empty string if no HRESULT was set.
     */
    public String getHresultDescription() {
        if (hresult == 0) return "";
        return HResult.describe(hresult);
    }

    @Override
    public String toString() {
        if (hresult != 0) {
            return super.toString() + " [HRESULT " + HResult.describe(hresult) + "]";
        }
        return super.toString();
    }
}
