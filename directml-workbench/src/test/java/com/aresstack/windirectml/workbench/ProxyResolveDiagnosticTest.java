package com.aresstack.windirectml.workbench;

import com.aresstack.winproxy.ProxyConfiguration;
import com.aresstack.winproxy.ProxyMode;
import com.aresstack.winproxy.ProxyResult;
import com.aresstack.winproxy.WindowsProxyResolver;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Integration diagnostic against the published win-proxy-java (beta.4) inside the
 * workbench runtime (JDK 21). Prints the result of the hardened-machine path
 * (PAC_URL_POWERSHELL, inline -Command) and the reg.exe path. No hard assertions.
 * <p>
 * Skipped automatically on non-Windows hosts (Linux/macOS CI runners) because
 * {@link WindowsProxyResolver} relies on Windows-specific APIs (PowerShell, registry).
 * </p>
 */
class ProxyResolveDiagnosticTest {

    private static final String TARGET = "https://plugins.gradle.org/m2/";

    @Test
    void reproduceWorkbenchProxyResolution() {
        assumeTrue(
                System.getProperty("os.name", "").toLowerCase().contains("windows"),
                "ProxyResolveDiagnosticTest requires Windows – skipped on non-Windows CI runners"
        );
        System.out.println("java.version = " + System.getProperty("java.version"));
        print(ProxyMode.PAC_URL_POWERSHELL);
        print(ProxyMode.PAC_URL_WINDOWS_SETTINGS);
    }

    private void print(ProxyMode mode) {
        ProxyConfiguration cfg = ProxyConfiguration.builder().mode(mode).build();
        ProxyResult result = new WindowsProxyResolver(cfg).resolve(TARGET);
        System.out.println("[" + mode + "] -> " + result + "  (kind=" + result.getKind() + ")");
    }
}
