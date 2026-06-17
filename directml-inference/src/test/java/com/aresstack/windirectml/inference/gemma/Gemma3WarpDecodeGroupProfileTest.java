package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyMath;
import com.aresstack.windirectml.inference.model.SafeTensorsReader;
import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorEntry;
import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorsFile;
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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-WARP-17: measured per-kernel-group timing breakdown of the WARP decode token, to choose the next
 * WARP bottleneck from data (not dispatch counts). Gated/heavy: {@code -Dgemma.warp.realModel=true
 * -Dgemma.warp.groupProfile=true}. Reports, per warm decode token, each group's calls/dispatches/uav
 * barriers, ms, ms/call and % of decode time, plus the real (coalesced) decode avg/token for reference.
 * Asserts the Paris smoke (" Paris", token 9079) on WARP.
 *
 * <p>The profiled pass runs synchronously (no coalescing), so its absolute ms/token is higher than the real
 * coalesced decode — the percentages are the deliverable. The normal runtime is unaffected.</p>
 */
@EnabledOnOs(OS.WINDOWS)
@EnabledIfSystemProperty(named = "gemma.warp.realModel", matches = "true")
class Gemma3WarpDecodeGroupProfileTest {

    private static final int[] FRANCE_IDS = {2, 818, 5279, 529, 7001, 563};
    private static final int EXPECTED_NEXT = 9079; // " Paris"
    private static final int WARMUP_TOKENS = 4;
    private static final int REF_TOKENS = 8;       // real coalesced decode reference
    private static final int PROFILE_TOKENS = 4;   // synchronous group-profiled tokens

    private static WindowsBindings wb;

    @BeforeAll
    static void initGpu() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML device");
        wb = new WindowsBindings();
        // WARP-focused slice: bind the explicit WARP software adapter (the CPU-only product path), not the
        // default DXGI adapter (which is a hardware GPU on this host).
        wb.init("directml", WindowsBindings.AdapterMode.WARP);
    }

    @AfterAll
    static void closeGpu() {
        if (wb != null) {
            wb.close();
        }
    }

    @Test
    void decodeGroupTimingBreakdown() throws Exception {
        assumeTrue(Boolean.getBoolean("gemma.warp.groupProfile"), "opt-in via -Dgemma.warp.groupProfile=true");
        Path dir = resolveModelDir();
        assumeTrue(dir != null, "Real Gemma 3 270M model not present");

        Gemma3Config config = new Gemma3ConfigReader().read(dir.resolve("config.json"));
        SafeTensorsFile file = SafeTensorsReader.read(dir.resolve("model.safetensors"));
        ByteBuffer embBb = decodeEmbeddingDirectFp32(file.tensors().get(Gemma3TensorNameMapper.EMBED_TOKENS));
        float[] finalNorm = Gemma3ReferenceWeights.decodeFloats(file.tensors().get(Gemma3TensorNameMapper.FINAL_NORM));
        Gemma3WarpLayerWeights[] layers = loadWarpLayers(file, config);
        Gemma3WarpWeights weights = Gemma3WarpWeights.ofByteBufferEmbedding(config, embBb, finalNorm, layers);

        try (Gemma3WarpDecodeSession sess = new Gemma3WarpDecodeSession(wb, weights)) {
            // Paris smoke (WARP) + warm the shaders.
            int first = DecoderOnlyMath.argmax(sess.prefillResidentBatched(FRANCE_IDS));
            assertEquals(EXPECTED_NEXT, first, "WARP prefill top-1 must be \" Paris\"");
            int tok = first;
            for (int i = 0; i < WARMUP_TOKENS; i++) {
                tok = sess.decodeNextTokenResident(tok);
            }

            // Real coalesced decode avg/token (no profiler attached — the product path).
            long t = System.nanoTime();
            for (int i = 0; i < REF_TOKENS; i++) {
                tok = sess.decodeNextTokenResident(tok);
            }
            double realDecodeMsPerToken = (System.nanoTime() - t) / 1e6 / REF_TOKENS;

            // Group-profiled (synchronous) decode tokens.
            WarpGroupProfiler profiler = new WarpGroupProfiler();
            sess.setGroupProfiler(profiler);
            try {
                for (int i = 0; i < PROFILE_TOKENS; i++) {
                    profiler.incTokens();
                    float[] logits = sess.decodeNextResident(tok);
                    long ts = System.nanoTime();
                    tok = DecoderOnlyMath.argmax(logits);
                    profiler.record("token-selection", 0, System.nanoTime() - ts);
                }
            } finally {
                sess.setGroupProfiler(null);
            }

            System.out.println("===== [GEMMA-WARP-17] WARP decode group timing =====");
            for (String line : profiler.render(realDecodeMsPerToken)) {
                System.out.println(line);
            }
            // Paris still holds after the profiled run (re-prefill is deterministic).
            assertEquals(EXPECTED_NEXT, sess.prefillNextTokenResidentBatched(FRANCE_IDS),
                    "WARP must still produce \" Paris\" after group profiling");
        }
    }

    // ── helpers (same loaders as the steady-state gated test) ──────────────────────────────

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
