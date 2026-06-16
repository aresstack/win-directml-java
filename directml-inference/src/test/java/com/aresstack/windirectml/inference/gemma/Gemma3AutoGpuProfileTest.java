package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-AUTO-GPU-1: compares the SAME native Gemma DirectML runtime on the explicit WARP adapter vs a
 * hardware (AUTO) adapter. Gated, heavy: {@code -Dgemma.warp.realModel=true -Dgemma.auto.gpu=true}
 * (profile via {@code -Ddirectml.generation.profile} is implied — the profile is always captured here). On a
 * host with no hardware GPU the HARDWARE leg skips with a clear message; WARP still runs and must produce
 * " Paris".
 */
@EnabledOnOs(OS.WINDOWS)
@EnabledIfSystemProperty(named = "gemma.warp.realModel", matches = "true")
class Gemma3AutoGpuProfileTest {

    private static final int MAX_NEW_TOKENS = 16;

    @Test
    void compareWarpVsHardwareAdapter() throws Exception {
        assumeTrue(Boolean.getBoolean("gemma.auto.gpu"), "opt-in via -Dgemma.auto.gpu=true");
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML device");
        Path dir = resolveModelDir();
        assumeTrue(dir != null, "Real Gemma 3 270M model not present");

        Path pkg = Files.createTempFile("gemma3-auto-gpu-", ".wdmlpack");
        try {
            Gemma3WdmlPackCompiler.compile(dir, pkg, true);
            Path tok = dir.resolve("tokenizer.json");

            String promptA = "The capital of France is";
            String promptB = "Berlin ist eine sehr große und lebendige Stadt mit vielen Museen und Parks und Menschen.";
            String promptC = promptB + " " + promptB + " " + promptB + " " + promptB; // ~longer prompt

            boolean warpRan = runOnAdapter("WARP", WindowsBindings.AdapterMode.WARP, pkg, tok,
                    promptA, promptB, promptC);
            boolean hwRan = runOnAdapter("HARDWARE", WindowsBindings.AdapterMode.HARDWARE, pkg, tok,
                    promptA, promptB, promptC);

            assertTrue(warpRan, "WARP adapter run must succeed");
            if (!hwRan) {
                System.out.println("[AUTO-GPU] HARDWARE adapter unavailable on this host — WARP-only result above.");
            }
        } finally {
            Files.deleteIfExists(pkg);
        }
    }

    /** Returns true if the adapter ran; false if it was a hardware adapter that is unavailable here. */
    private boolean runOnAdapter(String label, WindowsBindings.AdapterMode mode, Path pkg, Path tok,
                                 String promptA, String promptB, String promptC) throws IOException {
        System.out.println("===== [AUTO-GPU] adapter=" + label + " =====");
        Gemma3NativeWarpRuntime runtime = new Gemma3NativeWarpRuntime(pkg, tok, mode);
        // Smoke first: "The capital of France is" -> " Paris".
        Gemma3NativeWarpRuntime.Result smoke;
        try {
            smoke = runtime.generate(promptA, false, MAX_NEW_TOKENS);
        } catch (Exception e) {
            if (isNoHardware(e)) {
                System.out.println("[AUTO-GPU] " + label + " skipped: " + rootMessage(e));
                return false;
            }
            throw e;
        }
        assertTrue(smoke.text().contains("Paris"),
                label + " must produce \" Paris\" for \"The capital of France is\": '" + smoke.text() + "'");
        printProfile(label, "promptA(6)", smoke);

        printProfile(label, "promptB(~25)", runtime.generate(promptB, false, MAX_NEW_TOKENS));
        printProfile(label, "promptC(~100)", runtime.generate(promptC, false, MAX_NEW_TOKENS));
        return true;
    }

    private void printProfile(String adapter, String prompt, Gemma3NativeWarpRuntime.Result r) {
        Gemma3NativeWarpProfile p = r.profile();
        System.out.printf(java.util.Locale.ROOT,
                "[AUTO-GPU] %-8s %-12s adapter='%s' sw=%b promptTok=%d outTok=%d  prefill=%d ms"
                        + "  decode total=%d ms avg/token=%.1f ms  total=%d ms"
                        + "  submits/tok=%.1f fences/tok=%.1f readbacks/tok=%.1f dispatches/tok=%.1f uavBarriers/tok=%.1f%n",
                adapter, prompt, p.adapterDescription(), p.adapterSoftware(), p.promptTokens(), p.outputTokens(),
                p.prefillMs(), p.decodeTotalMs(), p.decodeAvgPerTokenMs(), p.runtimeTotalMs(),
                p.submitsPerToken(), p.fenceWaitsPerToken(), p.readbacksPerToken(),
                p.dispatchesPerToken(), p.uavBarriersPerToken());
    }

    private static boolean isNoHardware(Throwable e) {
        return rootMessage(e).toLowerCase().contains("no hardware");
    }

    private static String rootMessage(Throwable e) {
        Throwable t = e;
        while (t.getCause() != null) {
            t = t.getCause();
        }
        return t.getMessage() == null ? "" : t.getMessage();
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
