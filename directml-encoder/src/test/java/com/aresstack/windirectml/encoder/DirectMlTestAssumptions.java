package com.aresstack.windirectml.encoder;

import java.util.Locale;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

public final class DirectMlTestAssumptions {

    private DirectMlTestAssumptions() {
    }

    public static void skipIfHostDirectMlUnavailable(Throwable t) {
        String details = chainToString(t).toLowerCase(Locale.ROOT);
        if (details.contains("dxgi_error_device_removed")
                || details.contains("dxgi_error_device_reset")
                || details.contains("dxgi_error_device_hung")
                || details.contains("no directml device")
                || details.contains("directml requires windows")) {
            assumeTrue(false, "Skipping DirectML half: " + t.getMessage());
        }
    }

    private static String chainToString(Throwable t) {
        StringBuilder sb = new StringBuilder();
        Throwable cur = t;
        while (cur != null) {
            if (cur.getMessage() != null) {
                sb.append(cur.getMessage()).append('\n');
            }
            cur = cur.getCause();
        }
        return sb.toString();
    }
}
