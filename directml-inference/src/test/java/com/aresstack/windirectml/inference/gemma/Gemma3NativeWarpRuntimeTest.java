package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-WARP-11: native WARP runtime — the missing-package contract (device-free) and a gated real smoke
 * (compile a {@code .wdmlpack}, then generate " Paris" on the GPU).
 */
class Gemma3NativeWarpRuntimeTest {

    @Test
    void missingPackageMessageIsActionable() throws Exception {
        Path absent = Path.of("does", "not", "exist", "model_gemma3.wdmlpack");
        String msg = Gemma3NativeWarpRuntime.describeMissingPackage(absent);
        assertNotNull(msg, "missing package must produce a message");
        assertTrue(msg.contains("compiled .wdmlpack"), "message should mention the compiled package: " + msg);
        assertTrue(msg.contains("Download/Convert") || msg.contains("compiler"),
                "message should point to Download/Convert or the compiler: " + msg);

        Path present = Files.createTempFile("gemma-pkg-present", ".wdmlpack");
        try {
            assertNull(Gemma3NativeWarpRuntime.describeMissingPackage(present), "present package -> no message");
        } finally {
            Files.deleteIfExists(present);
        }
    }

    @EnabledOnOs(OS.WINDOWS)
    @Test
    void nativeWarpGeneratesParisFromCompiledPackage() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML device");
        Path dir = resolveModelDir();
        assumeTrue(dir != null, "Real Gemma 3 270M model not present");

        Path pkg = Files.createTempFile("gemma3-native-warp-", ".wdmlpack");
        try {
            Gemma3WdmlPackCompiler.compile(dir, pkg, true);
            assertNull(Gemma3NativeWarpRuntime.describeMissingPackage(pkg), "compiled package should be present");

            Gemma3NativeWarpRuntime runtime = new Gemma3NativeWarpRuntime(pkg, dir.resolve("tokenizer.json"));
            List<Integer> streamed = new ArrayList<>();
            // raw completion (no chat template) to match the validated "The capital of France is" -> " Paris."
            Gemma3NativeWarpRuntime.Result r = runtime.generate("The capital of France is", false, 3, streamed::add);

            System.out.println("[GEMMA-WARP-11] text='" + r.text() + "' promptTokens=" + r.promptTokens()
                    + " outputTokens=" + r.outputTokens() + " finish=" + r.finishReason());
            assertTrue(r.outputTokens() >= 1, "expected at least one generated token");
            assertEquals("WARP", r.backend(), "backend");
            assertTrue(r.text().contains("Paris"), "native WARP output should contain \"Paris\": '" + r.text() + "'");
            assertEquals(r.outputTokens(), streamed.size(), "streamed token count == output tokens");
        } finally {
            Files.deleteIfExists(pkg);
        }
    }

    private static Path resolveModelDir() {
        String override = System.getProperty("gemma.testModelDir");
        if (override != null && !override.isBlank()) {
            return dirIfValid(Path.of(override));
        }
        String appData = System.getenv("APPDATA");
        if (appData != null && !appData.isBlank()) {
            Path p = dirIfValid(Path.of(appData, ".directml", "model", "gemma-3-270m-it"));
            if (p != null) {
                return p;
            }
        }
        String home = System.getProperty("user.home");
        return home == null ? null : dirIfValid(Path.of(home, ".directml", "model", "gemma-3-270m-it"));
    }

    private static Path dirIfValid(Path dir) {
        return dir != null && Files.isRegularFile(dir.resolve("config.json"))
                && Files.isRegularFile(dir.resolve("model.safetensors"))
                && Files.isRegularFile(dir.resolve("tokenizer.json")) ? dir : null;
    }
}
