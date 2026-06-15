package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyMath;
import com.aresstack.windirectml.inference.model.SafeTensorsReader;
import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorEntry;
import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorsFile;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-WARP-10a: KV-cache decode session (prefill + single-token {@link Gemma3WarpDecodeSession#decodeNext})
 * vs the verified reference.
 *
 * <ul>
 *   <li><b>Synthetic</b>: prefill logits and one decodeNext step match the full-sequence CPU reference
 *       exactly (within tolerance), for a full-history config and a small-sliding-window config (so the
 *       local-layer window actually bites during decode); cache length grows correctly.</li>
 *   <li><b>Real model</b>: prefill top-1 stays " Paris" (9079), and decodeNext reproduces the
 *       full-recompute WARP forward pass for the next token (top-1) — the cache path == re-running the
 *       whole sequence, on the real Gemma 3 270M.</li>
 * </ul>
 *
 * <p>Skipped (assumption-aborted) without a DirectML/D3D12 device. The GPU computes in {@code float}.</p>
 */
@EnabledOnOs(OS.WINDOWS)
class Gemma3WarpDecodeSessionTest {

    private static final float ABS_TOL = 3e-3f;
    private static final float REL_TOL = 3e-3f;

    private static final int[] FRANCE_IDS = {2, 818, 5279, 529, 7001, 563};
    private static final int EXPECTED_NEXT = 9079; // " Paris"

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

    private static Gemma3Config config(int window) {
        // vocab=32, hidden=8, 4 layers, pattern 2 -> layers 1,3 full / 0,2 local; head_dim=4, GQA 2/1.
        return new Gemma3Config("gemma3_text", List.of("Gemma3ForCausalLM"),
                8, 16, 4, 2, 1, 4, 32, 32768, 1e-6, 1_000_000, 10_000, window, 2, List.of(),
                4, "gelu_pytorch_tanh", 2, 1, 0, true);
    }

    @Test
    void syntheticPrefillAndDecodeMatchReferenceFullHistory() throws Exception {
        assertDecodeParity(config(512), new Random(1001));
    }

    @Test
    void syntheticDecodeHonoursSlidingWindow() throws Exception {
        // window=2 so the local layers' visible range shrinks as the cache grows past the window.
        assertDecodeParity(config(2), new Random(1003));
    }

    @Test
    void cacheLengthGrowsWithPrefillAndDecode() throws Exception {
        Gemma3Config config = config(512);
        Gemma3ReferenceWeights ref = syntheticWeights(config, new Random(1007));
        int[] ids = {5, 11, 2, 28};
        try (Gemma3WarpDecodeSession sess = new Gemma3WarpDecodeSession(wb, Gemma3WarpWeights.from(ref))) {
            sess.prefill(ids);
            assertEquals(ids.length, sess.length(), "cache length after prefill");
            sess.decodeNext(13);
            assertEquals(ids.length + 1, sess.length(), "cache length after one decode");
            sess.decodeNext(4);
            assertEquals(ids.length + 2, sess.length(), "cache length after two decodes");
        }
    }

    @Test
    void realModelPrefillTop1IsParisAndDecodeMatchesFullRecompute() throws Exception {
        Path dir = resolveModelDir();
        assumeTrue(dir != null, "Real Gemma 3 270M model not present");
        Gemma3Config config = new Gemma3ConfigReader().read(dir.resolve("config.json"));
        assumeTrue(config.vocabSize() == 262144, "expected the real 270M-it config");
        SafeTensorsFile file = SafeTensorsReader.read(dir.resolve("model.safetensors"));

        ByteBuffer embBb = decodeEmbeddingDirectFp32(file.tensors().get(Gemma3TensorNameMapper.EMBED_TOKENS));
        float[] finalNorm = Gemma3ReferenceWeights.decodeFloats(file.tensors().get(Gemma3TensorNameMapper.FINAL_NORM));
        Gemma3WarpLayerWeights[] layers = loadWarpLayers(file, config);
        Gemma3WarpWeights weights = Gemma3WarpWeights.ofByteBufferEmbedding(config, embBb, finalNorm, layers);

        // Oracle: full-recompute WARP forward pass (already validated to produce " Paris").
        int ffNext;
        int[] withParis = append(FRANCE_IDS, EXPECTED_NEXT);
        try (Gemma3WarpForwardPass ff = new Gemma3WarpForwardPass(wb, weights)) {
            assertEquals(EXPECTED_NEXT, DecoderOnlyMath.argmax(ff.logitsForLastToken(FRANCE_IDS)),
                    "full forward prefill top-1 must be \" Paris\"");
            ffNext = DecoderOnlyMath.argmax(ff.logitsForLastToken(withParis));
        }

        // Decode session: prefill -> top-1 " Paris"; decodeNext(9079) must match the full recompute.
        try (Gemma3WarpDecodeSession sess = new Gemma3WarpDecodeSession(wb, weights)) {
            int prefillTop1 = sess.prefillNextToken(FRANCE_IDS);
            System.out.println("[GEMMA-WARP-DECODE] prefillTop1=" + prefillTop1 + " ffNext=" + ffNext);
            assertEquals(EXPECTED_NEXT, prefillTop1, "decode-session prefill top-1 must be \" Paris\"");
            int decodeTop1 = sess.decodeNextToken(EXPECTED_NEXT);
            assertEquals(ffNext, decodeTop1, "decodeNext top-1 must match the full-recompute forward pass");
        }
    }

    // ── helpers ──────────────────────────────────────────────────────

    private void assertDecodeParity(Gemma3Config config, Random rng) throws Exception {
        Gemma3ReferenceWeights ref = syntheticWeights(config, rng);
        Gemma3ReferenceForwardPass oracle = new Gemma3ReferenceForwardPass(ref);
        int[] ids = {7, 3, 19, 0, 25};

        float[] expectedPrefill = oracle.logitsForLastToken(ids);
        int next1 = DecoderOnlyMath.argmax(expectedPrefill);
        float[] expectedDecode = oracle.logitsForLastToken(append(ids, next1));
        int expectedDecodeTop1 = DecoderOnlyMath.argmax(expectedDecode);

        try (Gemma3WarpDecodeSession sess = new Gemma3WarpDecodeSession(wb, Gemma3WarpWeights.from(ref))) {
            float[] gpuPrefill = sess.prefill(ids);
            for (int o = 0; o < expectedPrefill.length; o++) {
                assertClose("prefill logits[" + o + "]", expectedPrefill[o], gpuPrefill[o]);
            }
            assertEquals(next1, DecoderOnlyMath.argmax(gpuPrefill), "prefill top-1");

            float[] gpuDecode = sess.decodeNext(next1);
            for (int o = 0; o < expectedDecode.length; o++) {
                assertClose("decode logits[" + o + "]", expectedDecode[o], gpuDecode[o]);
            }
            assertEquals(expectedDecodeTop1, DecoderOnlyMath.argmax(gpuDecode), "decodeNext top-1");
        }
    }

    private static int[] append(int[] a, int v) {
        int[] r = Arrays.copyOf(a, a.length + 1);
        r[a.length] = v;
        return r;
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

    private static Gemma3ReferenceWeights syntheticWeights(Gemma3Config c, Random rng) {
        int h = c.hiddenSize();
        int vocab = c.vocabSize();
        float[] embed = rand(rng, vocab * h, 0.1f);
        float[] finalNorm = rand(rng, h, 0.05f);
        Gemma3ReferenceWeights.Layer[] layers = new Gemma3ReferenceWeights.Layer[c.numHiddenLayers()];
        int d = c.headDim();
        int attnDim = c.attentionDim();
        int kvDim = c.keyValueDim();
        int inter = c.intermediateSize();
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Gemma3ReferenceWeights.Layer(
                    rand(rng, h, 0.05f),
                    rand(rng, attnDim * h, 0.05f), rand(rng, kvDim * h, 0.05f), rand(rng, kvDim * h, 0.05f),
                    rand(rng, h * attnDim, 0.05f),
                    rand(rng, d, 0.05f), rand(rng, d, 0.05f),
                    rand(rng, h, 0.05f), rand(rng, h, 0.05f),
                    rand(rng, inter * h, 0.05f), rand(rng, inter * h, 0.05f), rand(rng, h * inter, 0.05f),
                    rand(rng, h, 0.05f));
        }
        return new Gemma3ReferenceWeights(c, embed, finalNorm, layers);
    }

    private static float[] rand(Random rng, int n, float range) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (rng.nextFloat() * 2 - 1) * range;
        }
        return v;
    }

    private static void assertClose(String label, float want, float got) {
        float tol = ABS_TOL + REL_TOL * Math.abs(want);
        if (Math.abs(want - got) > tol) {
            fail(String.format("%s mismatch: want=%.6f got=%.6f diff=%.3e (tol=%.3e)",
                    label, want, got, Math.abs(want - got), tol));
        }
        assertEquals(want, got, tol, label);
    }
}
