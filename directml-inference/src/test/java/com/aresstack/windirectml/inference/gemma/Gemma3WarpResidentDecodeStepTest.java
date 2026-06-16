package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyMath;
import com.aresstack.windirectml.inference.model.SafeTensorsReader;
import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorEntry;
import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorsFile;
import com.aresstack.windirectml.inference.warp.WarpDenseProjection;
import com.aresstack.windirectml.runtime.DirectMlGpuBatch;
import com.aresstack.windirectml.windows.WarpExecutionContext;
import com.aresstack.windirectml.windows.WarpGpuBuffer;
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
 *
 * <p>GEMMA-WARP-13b-3b: the resident path additionally coalesces fences — each layer's pure compute
 * dispatches (incl. the resident projections) are submitted fire-and-forget under a {@code DirectMlGpuBatch}
 * and fenced once per layer drain, so {@code fenceWaits} fall far below {@code submits} while the output
 * stays identical. The deferred matvec is asserted bit-for-bit equal to the synchronous matvec.</p>
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
            // GEMMA-WARP-13c: the GPU-resident KV cache removes the per-layer k/v readbacks; only the final
            // logits are read back (1 per prefill), not 2×layers.
            assertTrue(residentRb <= 3, "resident prefill should read back only the final logits: " + residentRb);
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

    @Test
    void residentGpuKvCacheMatchesFloatPathAcrossGrowthAndWindow() throws Exception {
        // 13c: the GPU-resident KV cache matches the host float[] path across cache growth (initial cap 32)
        // and the sliding window (smallConfig slidingWindow=2 -> local layers windowed; full layers see the
        // whole history). Both paths are fed identical tokens so any divergence is a real cache bug.
        Gemma3Config config = smallConfig();
        Gemma3ReferenceWeights ref = syntheticWeights(config, new Random(13031));
        int[] prompt = {7, 3, 19, 0, 25};
        int decodeSteps = 40; // 5 + 40 = 45 positions > initial capacity 32 -> exercises growth

        try (Gemma3WarpDecodeSession sess = new Gemma3WarpDecodeSession(wb, Gemma3WarpWeights.from(ref))) {
            // float[] oracle: greedy ids
            int[] floatIds = new int[decodeSteps + 1];
            float[] fl = sess.prefill(prompt);
            floatIds[0] = DecoderOnlyMath.argmax(fl);
            for (int i = 1; i <= decodeSteps; i++) {
                fl = sess.decodeNext(floatIds[i - 1]);
                floatIds[i] = DecoderOnlyMath.argmax(fl);
            }

            // resident GPU-KV, fed the SAME tokens; argmax must agree at every step.
            WarpSubmissionStats.reset();
            WarpSubmissionStats.Snapshot r0 = WarpSubmissionStats.snapshot();
            float[] rl = sess.prefillResident(prompt);
            assertEquals(floatIds[0], DecoderOnlyMath.argmax(rl), "step 0 (prefill) top-1");
            for (int i = 1; i <= decodeSteps; i++) {
                rl = sess.decodeNextResident(floatIds[i - 1]);
                assertEquals(floatIds[i], DecoderOnlyMath.argmax(rl), "resident GPU-KV top-1 at step " + i);
            }
            long rb = WarpSubmissionStats.snapshot().minus(r0).readbacks();
            System.out.println("[13c] growth+window readbacks=" + rb + " over " + (decodeSteps + 1) + " logits calls");
            // ~1 readback per logits call (no per-layer k/v readbacks) -> bounded by the number of steps.
            assertTrue(rb <= decodeSteps + 2, "no per-layer readbacks expected: " + rb);
        }
    }

    @Test
    void syntheticBatchedPrefillMatchesTokenwiseResident() throws Exception {
        // 13e: batched prefill (whole prompt at once) == token-by-token resident prefill, with far fewer
        // submits. smallConfig has slidingWindow=2 so local layers are windowed + full layers see all
        // history; the prompt exercises both. Then decodeNext after batched prefill matches the tokenwise
        // path (the KV cache was filled to promptLen).
        Gemma3Config config = smallConfig();
        Gemma3ReferenceWeights ref = syntheticWeights(config, new Random(1305));
        int[] ids = {7, 3, 19, 0, 25, 11, 4};
        int feed = 9;

        try (Gemma3WarpDecodeSession sess = new Gemma3WarpDecodeSession(wb, Gemma3WarpWeights.from(ref))) {
            WarpSubmissionStats.Snapshot t0 = WarpSubmissionStats.snapshot();
            float[] tokenwise = sess.prefillResident(ids);
            long tokenwiseSubmits = WarpSubmissionStats.snapshot().minus(t0).submits();
            float[] tokenwiseNext = sess.decodeNextResident(feed);

            WarpSubmissionStats.Snapshot b0 = WarpSubmissionStats.snapshot();
            float[] batched = sess.prefillResidentBatched(ids);
            long batchedSubmits = WarpSubmissionStats.snapshot().minus(b0).submits();
            float[] batchedNext = sess.decodeNextResident(feed);

            for (int o = 0; o < tokenwise.length; o++) {
                assertClose("prefill logits[" + o + "]", tokenwise[o], batched[o]);
            }
            assertEquals(DecoderOnlyMath.argmax(tokenwise), DecoderOnlyMath.argmax(batched),
                    "batched prefill top-1 must match tokenwise resident prefill");
            for (int o = 0; o < tokenwiseNext.length; o++) {
                assertClose("decodeNext-after-prefill logits[" + o + "]", tokenwiseNext[o], batchedNext[o]);
            }
            assertEquals(DecoderOnlyMath.argmax(tokenwiseNext), DecoderOnlyMath.argmax(batchedNext),
                    "decodeNext after batched prefill must match decodeNext after tokenwise prefill");
            System.out.println("[13e] synthetic prefill submits: tokenwise=" + tokenwiseSubmits
                    + " batched=" + batchedSubmits + " (promptTokens=" + ids.length + ")");
            assertTrue(batchedSubmits < tokenwiseSubmits,
                    "batched prefill must submit fewer command lists: batched=" + batchedSubmits
                            + " tokenwise=" + tokenwiseSubmits);
        }
    }

    @Test
    void batchedResidentMatvecEqualsSyncMatvec() throws Exception {
        // 13b-3b: the deferred (batched) matvec must produce the exact same output as the synchronous one.
        int outN = 48;
        int inK = 64;
        Random rng = new Random(515);
        float[] w = rand(rng, outN * inK, 0.05f); // [N, K] row-major
        float[] x = rand(rng, inK, 0.5f);
        WarpExecutionContext ctx = new WarpExecutionContext(wb);
        try (WarpDenseProjection proj = WarpDenseProjection.fromDequantizedWeights(wb, "test.matvec", outN, inK, w)) {
            float[] sync;
            try (WarpGpuBuffer xb = ctx.upload(x); WarpGpuBuffer ob = proj.forwardResident(ctx, xb)) {
                sync = ob.readback(); // no batch active -> synchronous matvec path
            }
            float[] batched;
            try (WarpGpuBuffer xb = ctx.upload(x)) {
                try (DirectMlGpuBatch batch = DirectMlGpuBatch.begin(wb)) {
                    try (WarpGpuBuffer ob = proj.forwardResident(ctx, xb)) { // deferred matvec
                        batched = ob.readback();
                    }
                }
            }
            assertEquals(sync.length, batched.length, "matvec length");
            for (int i = 0; i < sync.length; i++) {
                assertEquals(sync[i], batched[i], 0f, "deferred matvec must equal sync matvec at [" + i + "]");
            }
        }
    }

    @Test
    void syntheticResidentPrefillCoalescesCommandLists() throws Exception {
        // 13d: resident prefill stays numerically identical to the float[] path, but the per-layer UAV
        // dispatches are coalesced into far fewer command-list submissions than the float path (which
        // submits one list per kernel).
        Gemma3Config config = smallConfig();
        Gemma3ReferenceWeights ref = syntheticWeights(config, new Random(343));
        int[] ids = {4, 11, 27, 2, 18};

        try (Gemma3WarpDecodeSession sess = new Gemma3WarpDecodeSession(wb, Gemma3WarpWeights.from(ref))) {
            WarpSubmissionStats.reset();
            WarpSubmissionStats.Snapshot f0 = WarpSubmissionStats.snapshot();
            float[] floatLogits = sess.prefill(ids);
            WarpSubmissionStats.Snapshot floatStats = WarpSubmissionStats.snapshot().minus(f0);

            WarpSubmissionStats.Snapshot r0 = WarpSubmissionStats.snapshot();
            float[] residentLogits = sess.prefillResident(ids);
            WarpSubmissionStats.Snapshot resStats = WarpSubmissionStats.snapshot().minus(r0);

            for (int o = 0; o < floatLogits.length; o++) {
                assertClose("prefill logits[" + o + "]", floatLogits[o], residentLogits[o]);
            }
            assertEquals(DecoderOnlyMath.argmax(floatLogits), DecoderOnlyMath.argmax(residentLogits), "top-1");
            System.out.println("[13d] synthetic prefill float=" + floatStats + " resident=" + resStats);
            // Command-list coalescing: the resident path submits far fewer command lists than the
            // per-kernel float path (UAV runs share one list; matvecs stay standalone for exact math).
            assertTrue(resStats.submits() < floatStats.submits(),
                    "resident submits must be coalesced below the float path: resident="
                            + resStats.submits() + " float=" + floatStats.submits());
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
            // 13e: token-by-token prefill (oracle) vs batched prefill (product path) — counters + parity.
            WarpSubmissionStats.Snapshot tp0 = WarpSubmissionStats.snapshot();
            int top1Tokenwise = sess.prefillNextTokenResident(FRANCE_IDS);
            WarpSubmissionStats.Snapshot tpd = WarpSubmissionStats.snapshot().minus(tp0);

            WarpSubmissionStats.Snapshot bp0 = WarpSubmissionStats.snapshot();
            int top1 = sess.prefillNextTokenResidentBatched(FRANCE_IDS);
            WarpSubmissionStats.Snapshot bpd = WarpSubmissionStats.snapshot().minus(bp0);
            System.out.println("[13e] prefill tokenwise=" + tpd + " batched=" + bpd
                    + " (promptTokens=" + FRANCE_IDS.length + ")");
            assertEquals(EXPECTED_NEXT, top1Tokenwise, "tokenwise prefill top-1 must be \" Paris\"");
            assertEquals(EXPECTED_NEXT, top1, "batched prefill top-1 must be \" Paris\"");
            assertTrue(bpd.submits() < tpd.submits(),
                    "batched prefill must submit fewer command lists: batched=" + bpd.submits()
                            + " tokenwise=" + tpd.submits());

            // measure one resident decode token after the batched prefill (cache filled to promptLen)
            WarpSubmissionStats.reset();
            WarpSubmissionStats.Snapshot d0 = WarpSubmissionStats.snapshot();
            int next = sess.decodeNextTokenResident(top1);
            WarpSubmissionStats.Snapshot dd = WarpSubmissionStats.snapshot().minus(d0);
            System.out.println("[13d] resident decode/token " + dd
                    + " (baselines: ~344 readbacks/token @13b-1, ~418 submits/token @13c); top1=" + top1 + " next=" + next);

            assertEquals(EXPECTED_NEXT, top1, "resident prefill top-1 must be \" Paris\"");
            // GEMMA-WARP-13c: GPU-resident KV cache -> only the final logits are read back per token.
            assertTrue(dd.readbacks() <= 3,
                    "resident decode readbacks/token must stay ~1 (logits only): " + dd.readbacks());
            // GEMMA-WARP-13d: command-list coalescing -> submits/token well below the ~418 of 13c.
            assertTrue(dd.submits() < 300,
                    "resident decode submits/token must be well below the ~418 13c baseline: " + dd.submits());
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
