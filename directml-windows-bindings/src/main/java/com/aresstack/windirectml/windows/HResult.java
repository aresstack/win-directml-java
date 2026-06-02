package com.aresstack.windirectml.windows;

import java.util.Map;

/**
 * Utility for dealing with Windows HRESULT codes.
 * <p>
 * Every Windows SDK function returns an {@code HRESULT} (a 32-bit signed int).
 * Bit 31 is the severity flag: 0 = success, 1 = error.
 * <p>
 * Includes a lookup table of well-known HRESULT codes from D3D12, DXGI,
 * and DirectML for human-readable diagnostics.
 */
public final class HResult {

    /**
     * S_OK – the universal "it worked" code.
     */
    public static final int S_OK = 0;

    // ── Well-known error HRESULTs ────────────────────────────────────────
    public static final int E_INVALIDARG = 0x80070057;
    public static final int E_OUTOFMEMORY = 0x8007000E;
    public static final int E_NOINTERFACE = 0x80004002;
    public static final int E_FAIL = 0x80004005;
    public static final int E_NOTIMPL = 0x80004001;
    public static final int E_ACCESSDENIED = 0x80070005;
    public static final int DXGI_ERROR_DEVICE_REMOVED = 0x887A0005;
    public static final int DXGI_ERROR_DEVICE_HUNG = 0x887A0006;
    public static final int DXGI_ERROR_DEVICE_RESET = 0x887A0007;
    public static final int DXGI_ERROR_DRIVER_INTERNAL_ERROR = 0x887A0020;
    public static final int DXGI_ERROR_NOT_FOUND = 0x887A0002;
    public static final int DXGI_ERROR_UNSUPPORTED = 0x887A0004;
    public static final int D3D12_ERROR_ADAPTER_NOT_FOUND = 0x887E0001;
    public static final int D3D12_ERROR_DRIVER_VERSION_MISMATCH = 0x887E0002;

    private static final Map<Integer, String> KNOWN_CODES = Map.ofEntries(
            Map.entry(S_OK, "S_OK"),
            Map.entry(E_INVALIDARG, "E_INVALIDARG"),
            Map.entry(E_OUTOFMEMORY, "E_OUTOFMEMORY"),
            Map.entry(E_NOINTERFACE, "E_NOINTERFACE"),
            Map.entry(E_FAIL, "E_FAIL"),
            Map.entry(E_NOTIMPL, "E_NOTIMPL"),
            Map.entry(E_ACCESSDENIED, "E_ACCESSDENIED"),
            Map.entry(DXGI_ERROR_DEVICE_REMOVED, "DXGI_ERROR_DEVICE_REMOVED"),
            Map.entry(DXGI_ERROR_DEVICE_HUNG, "DXGI_ERROR_DEVICE_HUNG"),
            Map.entry(DXGI_ERROR_DEVICE_RESET, "DXGI_ERROR_DEVICE_RESET"),
            Map.entry(DXGI_ERROR_DRIVER_INTERNAL_ERROR, "DXGI_ERROR_DRIVER_INTERNAL_ERROR"),
            Map.entry(DXGI_ERROR_NOT_FOUND, "DXGI_ERROR_NOT_FOUND"),
            Map.entry(DXGI_ERROR_UNSUPPORTED, "DXGI_ERROR_UNSUPPORTED"),
            Map.entry(D3D12_ERROR_ADAPTER_NOT_FOUND, "D3D12_ERROR_ADAPTER_NOT_FOUND"),
            Map.entry(D3D12_ERROR_DRIVER_VERSION_MISMATCH, "D3D12_ERROR_DRIVER_VERSION_MISMATCH")
    );

    private HResult() {
    }

    /**
     * {@code true} when the HRESULT indicates success (bit 31 clear).
     */
    public static boolean succeeded(int hr) {
        return hr >= 0;
    }

    /**
     * {@code true} when the HRESULT indicates failure (bit 31 set).
     */
    public static boolean failed(int hr) {
        return hr < 0;
    }

    /**
     * Throw {@link WindowsNativeException} if the HRESULT indicates failure.
     * The exception message includes the symbolic name (if known).
     *
     * @param hr   HRESULT value
     * @param call human-readable description of the call that produced it
     */
    public static void check(int hr, String call) throws WindowsNativeException {
        if (failed(hr)) {
            throw new WindowsNativeException(
                    String.format("%s failed: HRESULT %s", call, describe(hr)), hr);
        }
    }

    /**
     * Format an HRESULT as a hex string, e.g. {@code 0x80070057}.
     */
    public static String toHexString(int hr) {
        return String.format("0x%08X", hr);
    }

    /**
     * Return a human-readable description: symbolic name + hex.
     * E.g. {@code "E_INVALIDARG (0x80070057)"} or just {@code "0x887A0099"}.
     */
    public static String describe(int hr) {
        String name = KNOWN_CODES.get(hr);
        String hex = toHexString(hr);
        return name != null ? name + " (" + hex + ")" : hex;
    }
}

