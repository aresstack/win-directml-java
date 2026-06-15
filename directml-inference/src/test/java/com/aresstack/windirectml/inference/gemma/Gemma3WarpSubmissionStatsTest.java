package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyMath;
import com.aresstack.windirectml.windows.WarpSubmissionStats;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-WARP-13b-1: measure the real Gemma native-warp submit/fence/readback counts per prefill token and
 * per decode token (the "before" numbers for the later submit/fence batching). Opt-in
 * ({@code -Dgemma.warp.realModel=true}); prints a [13b-1] measurement log. No behaviour change is being
 * tested here — only that the counts are measurable and the output is still " Paris".
 */
@EnabledOnOs(OS.WINDOWS)
@EnabledIfSystemProperty(named = "gemma.warp.realModel", matches = "true")
class Gemma3WarpSubmissionStatsTest {

    private static final int[] FRANCE_IDS = {2, 818, 5279, 529, 7001, 563};
    private static final int EXPECTED_NEXT = 9079; // " Paris"

    @Test
    void measureSubmitsFencesReadbacksPerToken() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML device");
        Path dir = resolveModelDir();
        assumeTrue(dir != null, "Real Gemma 3 270M model not present");

        Path pkg = Files.createTempFile("gemma3-stats-", ".wdmlpack");
        try {
            Gemma3WdmlPackCompiler.compile(dir, pkg, true);
            Gemma3RuntimePackage rp = Gemma3RuntimePackage.open(pkg);
            Gemma3WarpWeights weights = rp.loadWarpWeightsHeapLight();

            WindowsBindings wb = new WindowsBindings();
            wb.init("directml");
            try (Gemma3WarpDecodeSession sess = new Gemma3WarpDecodeSession(wb, weights)) {
                int prompt = FRANCE_IDS.length;

                WarpSubmissionStats.reset();
                WarpSubmissionStats.Snapshot p0 = WarpSubmissionStats.snapshot();
                long t0 = System.nanoTime();
                float[] logits = sess.prefill(FRANCE_IDS);
                long prefillMs = (System.nanoTime() - t0) / 1_000_000L;
                WarpSubmissionStats.Snapshot pDelta = WarpSubmissionStats.snapshot().minus(p0);
                int top1 = DecoderOnlyMath.argmax(logits);

                WarpSubmissionStats.Snapshot d0 = WarpSubmissionStats.snapshot();
                long t1 = System.nanoTime();
                sess.decodeNext(top1);
                long decodeMs = (System.nanoTime() - t1) / 1_000_000L;
                WarpSubmissionStats.Snapshot dDelta = WarpSubmissionStats.snapshot().minus(d0);

                System.out.println("[13b-1] prefill promptTokens=" + prompt + " " + pDelta
                        + " wallMs=" + prefillMs
                        + " perPromptToken{submits=" + (pDelta.submits() / prompt)
                        + ", readbacks=" + (pDelta.readbacks() / prompt) + "}");
                System.out.println("[13b-1] decode(1 token) " + dDelta + " wallMs=" + decodeMs
                        + " decodeTokPerSec=" + String.format("%.2f", 1000.0 / Math.max(1, decodeMs)));

                assertEquals(EXPECTED_NEXT, top1, "prefill top-1 must still be \" Paris\"");
                assertTrue(pDelta.submits() > 0 && pDelta.readbacks() > 0, "prefill submits/readbacks measurable");
                assertTrue(dDelta.submits() > 0 && dDelta.readbacks() > 0, "decode submits/readbacks measurable");
            } finally {
                wb.close();
            }
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
