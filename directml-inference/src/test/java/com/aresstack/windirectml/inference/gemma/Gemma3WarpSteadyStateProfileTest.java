package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyMath;
import com.aresstack.windirectml.inference.model.SafeTensorsReader;
import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorEntry;
import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorsFile;
import com.aresstack.windirectml.windows.WarpSubmissionStats;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-WARP-13f: cold vs warm steady-state profiling of the Gemma native WARP resident product path
 * (batched prefill + resident decode). Gated (heavy, real model). Measures prefill and decode regions
 * separately, across prompt lengths, with an explicit {@link Gemma3WarpWarmup} between cold and warm so the
 * one-time lazy shader compile is visible in cold but excluded from warm. Runtimes are logged only (no
 * flaky ms assertions); the asserts are the " Paris" smoke, warm-up not changing token IDs, and counters
 * being present.
 */
@EnabledOnOs(OS.WINDOWS)
@EnabledIfSystemProperty(named = "gemma.warp.realModel", matches = "true")
class Gemma3WarpSteadyStateProfileTest {

    private static final int[] FRANCE_IDS = {2, 818, 5279, 529, 7001, 563};
    private static final int EXPECTED_NEXT = 9079; // " Paris"
    private static final int WARM_OUTPUT_TOKENS = 16;

    private static WindowsBindings wb;

    @BeforeAll
    static void initGpu() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML device");
        wb = new WindowsBindings();
        wb.init("directml");
    }

    @AfterAll
    static void closeGpu() {
        if (wb != null) {
            wb.close();
        }
    }

    @Test
    void coldVsWarmSteadyStateProfile() throws Exception {
        Path dir = resolveModelDir();
        assumeTrue(dir != null, "Real Gemma 3 270M model not present");
        Gemma3Config config = new Gemma3ConfigReader().read(dir.resolve("config.json"));
        SafeTensorsFile file = SafeTensorsReader.read(dir.resolve("model.safetensors"));
        ByteBuffer embBb = decodeEmbeddingDirectFp32(file.tensors().get(Gemma3TensorNameMapper.EMBED_TOKENS));
        float[] finalNorm = Gemma3ReferenceWeights.decodeFloats(file.tensors().get(Gemma3TensorNameMapper.FINAL_NORM));
        Gemma3WarpLayerWeights[] layers = loadWarpLayers(file, config);
        Gemma3WarpWeights weights = Gemma3WarpWeights.ofByteBufferEmbedding(config, embBb, finalNorm, layers);

        try (Gemma3WarpDecodeSession sess = new Gemma3WarpDecodeSession(wb, weights)) {
            // ── COLD: first ever batched prefill (includes lazy shader compile) + first decode ──
            WarpSubmissionStats.Snapshot c0 = WarpSubmissionStats.snapshot();
            long t = System.nanoTime();
            float[] coldLogits = sess.prefillResidentBatched(FRANCE_IDS);
            long coldPrefillMs = ms(t);
            WarpSubmissionStats.Snapshot coldPrefill = WarpSubmissionStats.snapshot().minus(c0);
            int coldTop1 = DecoderOnlyMath.argmax(coldLogits);

            WarpSubmissionStats.Snapshot cd0 = WarpSubmissionStats.snapshot();
            t = System.nanoTime();
            int coldNext = sess.decodeNextTokenResident(coldTop1);
            long coldDecodeMs = ms(t);
            WarpSubmissionStats.Snapshot coldDecode = WarpSubmissionStats.snapshot().minus(cd0);

            System.out.println("[13f] COLD prefill(6 tok)=" + coldPrefillMs + " ms " + coldPrefill
                    + "; first decode=" + coldDecodeMs + " ms " + coldDecode + "; top1=" + coldTop1 + " next=" + coldNext);
            assertEquals(EXPECTED_NEXT, coldTop1, "cold prefill top-1 must be \" Paris\"");

            // ── WARM-UP: compile all shaders/kernels via a throwaway short run ──
            int warmSteps = Gemma3WarpWarmup.warmUp(sess, Gemma3WarpWarmup.defaultWarmupPrompt(config), 4);
            assertTrue(warmSteps == 4, "warm-up should run the requested decode steps");
            // Warm-up must not change results: re-prefill France still gives " Paris".
            assertEquals(EXPECTED_NEXT, sess.prefillNextTokenResidentBatched(FRANCE_IDS),
                    "warm-up must not change token IDs");

            // ── WARM: prefill + 16-token decode for 6 / ~25 / ~100-token prompts ──
            warmProfile(sess, "6tok", FRANCE_IDS, config, true);
            warmProfile(sess, "25tok", buildPrompt(config, 25), config, false);
            warmProfile(sess, "100tok", buildPrompt(config, 100), config, false);
        }
    }

    private void warmProfile(Gemma3WarpDecodeSession sess, String label, int[] prompt, Gemma3Config config,
                             boolean expectParis) throws Exception {
        WarpSubmissionStats.Snapshot p0 = WarpSubmissionStats.snapshot();
        long t = System.nanoTime();
        float[] logits = sess.prefillResidentBatched(prompt);
        long prefillMs = ms(t);
        WarpSubmissionStats.Snapshot prefill = WarpSubmissionStats.snapshot().minus(p0);
        int first = DecoderOnlyMath.argmax(logits);
        if (expectParis) {
            assertEquals(EXPECTED_NEXT, first, "warm prefill top-1 must be \" Paris\" for " + label);
        }

        WarpSubmissionStats.Snapshot d0 = WarpSubmissionStats.snapshot();
        long td = System.nanoTime();
        int tok = first;
        for (int i = 0; i < WARM_OUTPUT_TOKENS; i++) {
            tok = sess.decodeNextTokenResident(tok);
        }
        long decodeTotalMs = ms(td);
        WarpSubmissionStats.Snapshot decode = WarpSubmissionStats.snapshot().minus(d0);

        double perTok = (double) decodeTotalMs / WARM_OUTPUT_TOKENS;
        System.out.printf("[13f] WARM %-6s prompt=%d  prefill=%d ms (submits=%d fences=%d readbacks=%d)"
                        + "  decode total=%d ms avg/token=%.1f ms"
                        + "  decode/token submits=%.1f fences=%.1f readbacks=%.1f%n",
                label, prompt.length, prefillMs, prefill.submits(), prefill.fenceWaits(), prefill.readbacks(),
                decodeTotalMs, perTok,
                decode.submits() / (double) WARM_OUTPUT_TOKENS,
                decode.fenceWaits() / (double) WARM_OUTPUT_TOKENS,
                decode.readbacks() / (double) WARM_OUTPUT_TOKENS);
        // counters present + readback hygiene (decode reads back only logits per token)
        assertTrue(prefill.submits() > 0, "prefill submits must be counted for " + label);
        assertTrue(decode.submits() > 0, "decode submits must be counted for " + label);
        assertTrue(decode.readbacks() <= WARM_OUTPUT_TOKENS + 2, "decode readbacks ~1/token for " + label);
    }

    // ── helpers ──────────────────────────────────────────────────────

    private static long ms(long startNanos) {
        return (System.nanoTime() - startNanos) / 1_000_000L;
    }

    /** A valid prompt of {@code n} tokens (bos + arbitrary in-range ids); content irrelevant for timing. */
    private static int[] buildPrompt(Gemma3Config config, int n) {
        int vocab = config.vocabSize();
        int[] ids = new int[n];
        ids[0] = config.bosTokenId();
        for (int i = 1; i < n; i++) {
            ids[i] = 1 + Math.floorMod(i * 131 + 7, vocab - 1);
        }
        return ids;
    }

    private static Gemma3WarpLayerWeights[] loadWarpLayers(SafeTensorsFile f, Gemma3Config c) throws IOException {
        int n = c.numHiddenLayers();
        Gemma3WarpLayerWeights[] out = new Gemma3WarpLayerWeights[n];
        for (int i = 0; i < n; i++) {
            out[i] = new Gemma3WarpLayerWeights(
                    floats(f, Gemma3TensorNameMapper.inputLayerNorm(i)),
                    floats(f, Gemma3TensorNameMapper.qProj(i)), floats(f, Gemma3TensorNameMapper.kProj(i)),
                    floats(f, Gemma3TensorNameMapper.vProj(i)), floats(f, Gemma3TensorNameMapper.oProj(i)),
                    floats(f, Gemma3TensorNameMapper.qNorm(i)), floats(f, Gemma3TensorNameMapper.kNorm(i)),
                    floats(f, Gemma3TensorNameMapper.postAttentionLayerNorm(i)),
                    floats(f, Gemma3TensorNameMapper.preFeedforwardLayerNorm(i)),
                    floats(f, Gemma3TensorNameMapper.gateProj(i)), floats(f, Gemma3TensorNameMapper.upProj(i)),
                    floats(f, Gemma3TensorNameMapper.downProj(i)),
                    floats(f, Gemma3TensorNameMapper.postFeedforwardLayerNorm(i)));
        }
        return out;
    }

    private static float[] floats(SafeTensorsFile f, String name) throws IOException {
        SafeTensorEntry e = f.tensors().get(name);
        if (e == null) {
            throw new IOException("missing tensor: " + name);
        }
        return Gemma3ReferenceWeights.decodeFloats(e);
    }

    private static ByteBuffer decodeEmbeddingDirectFp32(SafeTensorEntry e) {
        long count = 1;
        for (long d : e.shape()) {
            count *= d;
        }
        int n = Math.toIntExact(count);
        ByteBuffer out = ByteBuffer.allocateDirect(n * 4).order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer src = e.dataBuffer();
        switch (e.dtype()) {
            case "F32" -> {
                for (int i = 0; i < n; i++) {
                    out.putFloat(src.getFloat());
                }
            }
            case "F16" -> {
                for (int i = 0; i < n; i++) {
                    out.putFloat(Float.float16ToFloat(src.getShort()));
                }
            }
            case "BF16" -> {
                for (int i = 0; i < n; i++) {
                    out.putFloat(Float.intBitsToFloat((src.getShort() & 0xFFFF) << 16));
                }
            }
            default -> throw new IllegalArgumentException("unsupported embedding dtype: " + e.dtype());
        }
        out.flip();
        return out;
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
                && Files.isRegularFile(dir.resolve("model.safetensors")) ? dir : null;
    }
}
