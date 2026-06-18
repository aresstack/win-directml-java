package com.aresstack.windirectml.inference.phi3;

import com.aresstack.windirectml.inference.model.RuntimeTensorCatalog;
import com.aresstack.windirectml.inference.phi3.Phi3Weights.LayerWeights;
import com.aresstack.windirectml.inference.phi3.Phi3Weights.QuantizedWeight;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * PHI3-WDMLPACK-COMPILER-1: structural round-trip for the Phi-3 wdmlpack compiler + package loader.
 *
 * <p>The non-gated test builds a tiny synthetic {@link Phi3Weights} (no ONNX), compiles it through
 * {@link Phi3WdmlPackCompiler#writePackage}, reopens it with {@link Phi3RuntimePackage}, and asserts the config and
 * every tensor (INT4 triplets + fp32 vectors) round-trip byte-exactly. The real Phi-3-mini ONNX compile is gated
 * ({@code -Dphi3.compile.realModel=true}) because it needs the ~2.3 GB model.</p>
 */
class Phi3WdmlPackCompilerTest {

    @TempDir
    Path tempDir;

    @Test
    void syntheticWeightsRoundTripThroughPackage() throws Exception {
        Phi3Config config = new Phi3Config(4, 2, 1, 2, 6, 8, 8, 1e-5f, 10000.0f);
        Phi3Weights original = syntheticWeights(config);

        Path output = tempDir.resolve("model_phi3.wdmlpack");
        Phi3WdmlPackCompiler.Phi3CompileResult result =
                Phi3WdmlPackCompiler.writePackage(config, original, output, true);

        assertTrue(Files.isRegularFile(output), "package must be written");
        assertEquals(1, result.layers());
        assertTrue(result.tensorCount() > 0);
        assertTrue(result.payloadBytes() > 0);

        Phi3RuntimePackage pkg = Phi3RuntimePackage.open(output);
        assertEquals("phi3-native-payload", pkg.runtimeLoadMode());
        Phi3Config reloadedConfig = pkg.config();
        assertEquals(config, reloadedConfig);

        Phi3Weights reloaded = pkg.weights();
        assertArrayEquals(original.embedTokens, reloaded.embedTokens, 0.0f);
        assertArrayEquals(original.cosCache, reloaded.cosCache, 0.0f);
        assertArrayEquals(original.sinCache, reloaded.sinCache, 0.0f);
        assertArrayEquals(original.finalNormWeight, reloaded.finalNormWeight, 0.0f);
        assertQuantizedEquals(original.lmHead, reloaded.lmHead);

        assertEquals(original.layers.length, reloaded.layers.length);
        for (int l = 0; l < original.layers.length; l++) {
            LayerWeights a = original.layers[l];
            LayerWeights b = reloaded.layers[l];
            assertArrayEquals(a.inputNormWeight(), b.inputNormWeight(), 0.0f);
            assertArrayEquals(a.postNormWeight(), b.postNormWeight(), 0.0f);
            assertArrayEquals(a.attnOutScale(), b.attnOutScale(), 0.0f);
            assertArrayEquals(a.mlpOutScale(), b.mlpOutScale(), 0.0f);
            assertQuantizedEquals(a.qProj(), b.qProj());
            assertQuantizedEquals(a.kProj(), b.kProj());
            assertQuantizedEquals(a.vProj(), b.vProj());
            assertQuantizedEquals(a.oProj(), b.oProj());
            assertQuantizedEquals(a.gateUpProj(), b.gateUpProj());
            assertQuantizedEquals(a.downProj(), b.downProj());
        }
        reloaded.close();
    }

    @Test
    @EnabledIf("realModelCompileEnabled")
    void realPhi3MiniOnnxCompilesAndReopens() throws Exception {
        Path modelDir = resolveRealModelDir();
        assertNotNull(modelDir, "real Phi-3 model dir must be present when -Dphi3.compile.realModel=true");
        Path output = tempDir.resolve("model_phi3.wdmlpack");

        // Streaming compile (COMPILER-2): tensors are copied/converted straight from the mmap'd model.onnx.data,
        // so this runs within a small heap (no full ~2.4 GB Phi3Weights on the heap).
        Phi3WdmlPackCompiler.Phi3CompileResult result =
                Phi3WdmlPackCompiler.compile(new Phi3CompileOptions(modelDir, output, true));
        assertTrue(Files.isRegularFile(output));
        System.out.println("[PHI3-COMPILE] tensors=" + result.tensorCount()
                + " payloadBytes=" + result.payloadBytes() + " layers=" + result.layers()
                + " wdmlpackBytes=" + Files.size(output));

        // The heap-safe streaming compile is the slice goal: a real ~2.4 GB package is produced within a small heap.
        Phi3Config cfg = Phi3Config.load(modelDir.resolve("config.json"));
        // 4 globals (embed/cos/sin/final) + lm_head triplet (3) + per-layer (4 fp32 + 6 quant*3) = 7 + 22*L.
        assertEquals(7 + 22 * cfg.numHiddenLayers(), result.tensorCount(), "expected full Phi-3 role-tensor count");
        assertTrue(result.payloadBytes() > 0);

        // WDMLPACK-LARGE-READER-1: the shared reader now handles >2 GB packages (positional manifest read +
        // per-tensor windowed mapping), so the real ~2.39 GB Phi-3-mini package reloads. Structural check only via the
        // mmap-backed catalog (full weights() reconstruction is a separate runtime-memory concern).
        assertTrue(Files.size(output) > Integer.MAX_VALUE, "real Phi-3-mini package is expected to exceed 2 GB");
        Phi3RuntimePackage pkg = Phi3RuntimePackage.open(output);
        assertEquals(cfg, pkg.config(), "config must round-trip from the >2GB package manifest");
        RuntimeTensorCatalog catalog = pkg.runtimeTensorCatalog();
        assertArrayEquals(new long[]{cfg.vocabSize(), cfg.hiddenSize()},
                catalog.get(Phi3WdmlPackRoles.EMBED_TOKENS).dims(), "embed_tokens dims");
        assertTrue(catalog.contains(Phi3WdmlPackRoles.qweight(
                        Phi3WdmlPackRoles.layer(0, Phi3WdmlPackRoles.Q_PROJ))),
                "package must contain layer 0 q_proj qweight");
        assertEquals(7 + 22 * cfg.numHiddenLayers(), catalog.size(), "expected full Phi-3 role-tensor count");
    }

    // PHI3-WORKBENCH-RUNNABLE-1: a full package->weights()->Phi3Runtime decode smoke is intentionally NOT committed
    // here. The eager weights() reconstruction needs ~3 GB heap and, together with the 2.39 GB package mmap,
    // over-commits the host's free RAM inside the forked test JVM (the JVM is OS-killed, not a clean failure). The
    // runtime decode smoke + the PLANNED->EXPERIMENTAL status flip are deferred to a heap-light Phi runtime-package
    // loader slice (decision C). See docs/phi3-wdmlpack-compiler-plan.md.

    // ── helpers ──────────────────────────────────────────────────────────

    private static Phi3Weights syntheticWeights(Phi3Config c) {
        int h = c.hiddenSize();
        int inter = c.intermediateSize();
        float[] embed = ramp(c.vocabSize() * h, 0.1f);
        float[] cos = ramp(8, 0.2f);
        float[] sin = ramp(8, 0.3f);
        float[] finalNorm = ramp(h, 0.4f);
        QuantizedWeight lmHead = quant(c.vocabSize(), h, 4, 1);

        LayerWeights layer = new LayerWeights(
                ramp(h, 0.5f),                 // inputNorm
                quant(h, h, 4, 2),             // qProj
                quant(h, h, 4, 3),             // kProj
                quant(h, h, 4, 4),             // vProj
                ramp(h, 0.6f),                 // attnOutScale [hidden]
                quant(h, h, 4, 5),             // oProj
                ramp(h, 0.7f),                 // postNorm
                quant(2 * inter, h, 4, 6),     // gateUpProj
                ramp(inter, 0.8f),             // mlpOutScale [intermediate]
                quant(h, inter, 4, 7));        // downProj
        return Phi3Weights.ofRecords(c, embed, cos, sin, new LayerWeights[]{layer}, finalNorm, lmHead);
    }

    private static float[] ramp(int n, float step) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = i * step - 1.0f;
        }
        return out;
    }

    private static QuantizedWeight quant(int n, int k, int blockSize, int seed) {
        int blocksPerRow = k / blockSize;
        byte[] qWeight = new byte[n * blocksPerRow * (blockSize / 2)];
        for (int i = 0; i < qWeight.length; i++) {
            qWeight[i] = (byte) ((i * 7 + seed) & 0xFF);
        }
        float[] scales = new float[n * blocksPerRow];
        for (int i = 0; i < scales.length; i++) {
            scales[i] = (i + seed) * 0.01f;
        }
        byte[] zp = new byte[(n * blocksPerRow + 1) / 2];
        for (int i = 0; i < zp.length; i++) {
            zp[i] = (byte) ((i * 3 + seed) & 0xFF);
        }
        return new QuantizedWeight(qWeight, scales, zp, n, k, blockSize);
    }

    private static void assertQuantizedEquals(QuantizedWeight a, QuantizedWeight b) {
        assertEquals(a.N(), b.N(), "N");
        assertEquals(a.K(), b.K(), "K");
        assertEquals(a.blockSize(), b.blockSize(), "blockSize");
        assertArrayEquals(a.qWeight(), b.qWeight(), "qWeight bytes must round-trip exactly (INT4 preserved)");
        assertArrayEquals(a.scales(), b.scales(), 0.0f, "scales");
        assertArrayEquals(a.zeroPoints(), b.zeroPoints(), "zeroPoints bytes");
    }

    @SuppressWarnings("unused")
    static boolean realModelCompileEnabled() {
        return Boolean.getBoolean("phi3.compile.realModel") && resolveRealModelDir() != null;
    }

    static Path resolveRealModelDir() {
        String override = System.getProperty("phi3.testModelDir");
        if (override != null && !override.isBlank() && Files.isRegularFile(Path.of(override, "model.onnx"))) {
            return Path.of(override);
        }
        for (String rel : new String[]{"phi-3-mini-4k-instruct-onnx", "microsoft/Phi-3-mini-4k-instruct-onnx"}) {
            for (Path base : new Path[]{Path.of("model", rel), Path.of("../model", rel)}) {
                if (Files.isRegularFile(base.resolve("model.onnx"))) {
                    return base;
                }
            }
        }
        return null;
    }
}
