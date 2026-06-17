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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-BF16-PACK-2: end-to-end check that retaining the tied embedding/LM head as BF16 (the new product
 * load path) is byte-equivalent to the FP32 path, gives " Paris" (9079), halves the retained host bytes, and
 * does not slow decode. Gated: {@code -Dgemma.warp.realModel=true -Dgemma.warp.bf16Probe=true}, WARP adapter.
 */
@EnabledOnOs(OS.WINDOWS)
@EnabledIfSystemProperty(named = "gemma.warp.realModel", matches = "true")
class Gemma3Bf16EmbeddingLoadProbeTest {

    private static final int[] FRANCE_IDS = {2, 818, 5279, 529, 7001, 563};
    private static final int EXPECTED_NEXT = 9079; // " Paris"
    private static final int DECODE_TOKENS = 8;

    private static WindowsBindings wb;

    @BeforeAll
    static void initGpu() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML device");
        wb = new WindowsBindings();
        wb.init("directml", WindowsBindings.AdapterMode.WARP);
    }

    @AfterAll
    static void closeGpu() {
        if (wb != null) {
            wb.close();
        }
    }

    @Test
    void bf16EmbeddingMatchesFp32AndKeepsParis() throws Exception {
        assumeTrue(Boolean.getBoolean("gemma.warp.bf16Probe"), "opt-in via -Dgemma.warp.bf16Probe=true");
        Path dir = resolveModelDir();
        assumeTrue(dir != null, "Real Gemma 3 270M model not present");

        Gemma3Config config = new Gemma3ConfigReader().read(dir.resolve("config.json"));
        int vocab = config.vocabSize();
        int hidden = config.hiddenSize();
        float scale = (float) config.embeddingScale();
        SafeTensorsFile file = SafeTensorsReader.read(dir.resolve("model.safetensors"));
        SafeTensorEntry embEntry = entry(file, Gemma3TensorNameMapper.EMBED_TOKENS);
        float[] finalNorm = Gemma3ReferenceWeights.decodeFloats(entry(file, Gemma3TensorNameMapper.FINAL_NORM));
        Gemma3WarpLayerWeights[] layers = loadWarpLayers(file, config);

        // ── Load: FP32 decode vs BF16 copy (the embedding block), with timings + retained bytes ──
        long t0 = System.nanoTime();
        ByteBuffer embFp32 = Gemma3WeightBufferView.decodeFp32LittleEndian(embEntry);
        long fp32LoadMs = (System.nanoTime() - t0) / 1_000_000L;
        t0 = System.nanoTime();
        Gemma3Bf16WeightView embBf16 = Gemma3Bf16WeightView.ofBf16Copy(embEntry);
        long bf16LoadMs = (System.nanoTime() - t0) / 1_000_000L;

        Gemma3WarpWeights fp32Weights = Gemma3WarpWeights.ofByteBufferEmbedding(config, embFp32, finalNorm, layers);
        Gemma3WarpWeights bf16Weights = Gemma3WarpWeights.ofBf16Embedding(config, embBf16, finalNorm, layers);

        // ── Row-decode equality for the required token ids ──
        int[] ids = {0, 1, EXPECTED_NEXT, vocab - 1, new Random(7).nextInt(vocab)};
        for (int id : ids) {
            float[] fp32Row = fp32Weights.embedScaled(new int[]{id}, scale)[0];
            float[] bf16Row = bf16Weights.embedScaled(new int[]{id}, scale)[0];
            assertArrayEquals(fp32Row, bf16Row, 0f, "embedding row for id " + id + " must match exactly");
        }

        // ── Paris + decode timing on each path (FP32 first, then BF16; sequential to bound memory) ──
        double fp32DecodeMs = parisAndDecode(fp32Weights, "fp32");
        double bf16DecodeMs = parisAndDecode(bf16Weights, "bf16");

        long fp32Retained = (long) vocab * hidden * Float.BYTES;
        long bf16Retained = embBf16.retainedBytes();
        double mb = 1024.0 * 1024.0;
        System.out.println("===== [GEMMA-BF16-PACK-2] tied embedding/LM-head host memory =====");
        System.out.printf(Locale.ROOT, "retained host bytes (embedding): FP32 %.1f MB -> BF16 %.1f MB (save %.1f MB)%n",
                fp32Retained / mb, bf16Retained / mb, (fp32Retained - bf16Retained) / mb);
        System.out.printf(Locale.ROOT, "embedding decode/copy at load: FP32 %d ms -> BF16 %d ms%n", fp32LoadMs, bf16LoadMs);
        System.out.printf(Locale.ROOT, "decode avg/token: FP32 %.1f ms -> BF16 %.1f ms%n", fp32DecodeMs, bf16DecodeMs);

        // Decode must not get relevantly slower (generous bound; the per-row inflate is tiny vs the GEMMs).
        assertTrue(bf16DecodeMs <= fp32DecodeMs * 1.5 + 25,
                "BF16 decode must not be relevantly slower: fp32=" + fp32DecodeMs + " bf16=" + bf16DecodeMs);
    }

    private double parisAndDecode(Gemma3WarpWeights weights, String label) throws Exception {
        try (Gemma3WarpDecodeSession sess = new Gemma3WarpDecodeSession(wb, weights)) {
            int tok = DecoderOnlyMath.argmax(sess.prefillResidentBatched(FRANCE_IDS));
            assertEquals(EXPECTED_NEXT, tok, label + " path must give \" Paris\"");
            for (int i = 0; i < 2; i++) {
                tok = sess.decodeNextTokenResident(tok); // warm
            }
            long t = System.nanoTime();
            for (int i = 0; i < DECODE_TOKENS; i++) {
                tok = sess.decodeNextTokenResident(tok);
            }
            return (System.nanoTime() - t) / 1e6 / DECODE_TOKENS;
        }
    }

    private static void assertTrue(boolean cond, String msg) {
        org.junit.jupiter.api.Assertions.assertTrue(cond, msg);
    }

    private static SafeTensorEntry entry(SafeTensorsFile f, String name) throws IOException {
        SafeTensorEntry e = f.tensors().get(name);
        if (e == null) {
            throw new IOException("missing tensor: " + name);
        }
        return e;
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
        return Gemma3ReferenceWeights.decodeFloats(entry(f, name));
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
