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
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-WARP-13b-3a: the resident decode path matches the synchronous float[] path numerically and does
 * far fewer readbacks per token (intermediates stay GPU-resident; only the new k/v and the final logits
 * are read back).
 */
@EnabledOnOs(OS.WINDOWS)
class Gemma3WarpResidentDecodeStepTest {

    private static final float ABS_TOL = 2e-3f;
    private static final float REL_TOL = 2e-3f;
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

    private static Gemma3Config smallConfig() {
        return new Gemma3Config("gemma3_text", List.of("Gemma3ForCausalLM"),
                8, 16, 4, 2, 1, 4, 32, 32768, 1e-6, 1_000_000, 10_000, 2, 2, List.of(),
                4, "gelu_pytorch_tanh", 2, 1, 0, true);
    }

    @Test
    void syntheticResidentPrefillMatchesFloatPathWithFewerReadbacks() throws Exception {
        Gemma3Config config = smallConfig();
        Gemma3ReferenceWeights ref = syntheticWeights(config, new Random(331));
        int[] ids = {7, 3, 19, 0, 25};

        try (Gemma3WarpDecodeSession sess = new Gemma3WarpDecodeSession(wb, Gemma3WarpWeights.from(ref))) {
            WarpSubmissionStats.reset();
            WarpSubmissionStats.Snapshot f0 = WarpSubmissionStats.snapshot();
            float[] floatLogits = sess.prefill(ids);
            long floatRb = WarpSubmissionStats.snapshot().minus(f0).readbacks();

            WarpSubmissionStats.Snapshot r0 = WarpSubmissionStats.snapshot();
            float[] residentLogits = sess.prefillResident(ids);
            long residentRb = WarpSubmissionStats.snapshot().minus(r0).readbacks();

            for (int o = 0; o < floatLogits.length; o++) {
                assertClose("prefill logits[" + o + "]", floatLogits[o], residentLogits[o]);
            }
            assertEquals(DecoderOnlyMath.argmax(floatLogits), DecoderOnlyMath.argmax(residentLogits), "top-1");
            System.out.println("[13b-3] synthetic prefill floatReadbacks=" + floatRb
                    + " residentReadbacks=" + residentRb + " (promptTokens=" + ids.length + ")");
            assertTrue(residentRb < floatRb, "resident must do fewer readbacks: resident="
                    + residentRb + " float=" + floatRb);
        }
    }

    @Test
    void syntheticResidentDecodeNextMatchesFloatPath() throws Exception {
        Gemma3Config config = smallConfig();
        Gemma3ReferenceWeights ref = syntheticWeights(config, new Random(337));
        int[] ids = {1, 14, 30, 8};
        int feed = 5;

        try (Gemma3WarpDecodeSession sess = new Gemma3WarpDecodeSession(wb, Gemma3WarpWeights.from(ref))) {
            sess.prefill(ids);
            float[] floatNext = sess.decodeNext(feed);

            sess.prefillResident(ids); // resets cache, resident
            float[] residentNext = sess.decodeNextResident(feed);

            for (int o = 0; o < floatNext.length; o++) {
                assertClose("decodeNext logits[" + o + "]", floatNext[o], residentNext[o]);
            }
            assertEquals(DecoderOnlyMath.argmax(floatNext), DecoderOnlyMath.argmax(residentNext), "decode top-1");
        }
    }

    @EnabledIfSystemProperty(named = "gemma.warp.realModel", matches = "true")
    @Test
    void realModelResidentParisAndReadbacksWellBelowBaseline() throws Exception {
        Path dir = resolveModelDir();
        assumeTrue(dir != null, "Real Gemma 3 270M model not present");
        Gemma3Config config = new Gemma3ConfigReader().read(dir.resolve("config.json"));
        SafeTensorsFile file = SafeTensorsReader.read(dir.resolve("model.safetensors"));
        ByteBuffer embBb = decodeEmbeddingDirectFp32(file.tensors().get(Gemma3TensorNameMapper.EMBED_TOKENS));
        float[] finalNorm = Gemma3ReferenceWeights.decodeFloats(file.tensors().get(Gemma3TensorNameMapper.FINAL_NORM));
        Gemma3WarpLayerWeights[] layers = loadWarpLayers(file, config);
        Gemma3WarpWeights weights = Gemma3WarpWeights.ofByteBufferEmbedding(config, embBb, finalNorm, layers);

        try (Gemma3WarpDecodeSession sess = new Gemma3WarpDecodeSession(wb, weights)) {
            int top1 = sess.prefillNextTokenResident(FRANCE_IDS);
            // measure one resident decode token
            WarpSubmissionStats.reset();
            WarpSubmissionStats.Snapshot d0 = WarpSubmissionStats.snapshot();
            int next = sess.decodeNextTokenResident(top1);
            WarpSubmissionStats.Snapshot dd = WarpSubmissionStats.snapshot().minus(d0);
            System.out.println("[13b-3] resident decode/token " + dd
                    + " (13b-1 baseline ~344 readbacks/token); top1=" + top1 + " next=" + next);

            assertEquals(EXPECTED_NEXT, top1, "resident prefill top-1 must be \" Paris\"");
            assertTrue(dd.readbacks() < 100,
                    "resident decode readbacks/token must be well below the ~344 baseline: " + dd.readbacks());
        }
    }

    // ── helpers ──────────────────────────────────────────────────────

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
                && Files.isRegularFile(dir.resolve("model.safetensors"))
                && Files.isRegularFile(dir.resolve("tokenizer.json")) ? dir : null;
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
        assertEquals(want, got, tol, label);
    }
}
