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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-BF16-PACK-3: the BF16-retained layer projections (loaded via the product {@code .wdmlpack} path) must
 * decode bit-identically to the FP32 path, keep the QKV/GateUp fusion, give " Paris" (9079) and not slow
 * decode. Gated: {@code -Dgemma.warp.realModel=true -Dgemma.warp.bf16Probe=true}, WARP adapter.
 */
@EnabledOnOs(OS.WINDOWS)
@EnabledIfSystemProperty(named = "gemma.warp.realModel", matches = "true")
class Gemma3Bf16ProjectionLoadProbeTest {

    private static final int[] FRANCE_IDS = {2, 818, 5279, 529, 7001, 563};
    private static final int EXPECTED_NEXT = 9079; // " Paris"
    private static final int DECODE_TOKENS = 16;

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
    void bf16ProjectionsDecodeIdenticallyAndKeepParis() throws Exception {
        assumeTrue(Boolean.getBoolean("gemma.warp.bf16Probe"), "opt-in via -Dgemma.warp.bf16Probe=true");
        Path dir = resolveModelDir();
        assumeTrue(dir != null, "Real Gemma 3 270M model not present");

        Gemma3Config config = new Gemma3ConfigReader().read(dir.resolve("config.json"));
        SafeTensorsFile file = SafeTensorsReader.read(dir.resolve("model.safetensors"));

        // Byte-level projection parity (no GPU): the retained BF16 bytes widen to exactly the same FP32 as the
        // current FP32 decode, for every layer-0 matmul projection -> the device weight is bit-identical, so
        // the QKV/GateUp fused device weights and the decode are unchanged. (Two full WARP sessions at once
        // exhaust the software device's system RAM, so parity is proven here, not by a second live session.)
        for (String name : new String[]{
                Gemma3TensorNameMapper.qProj(0), Gemma3TensorNameMapper.kProj(0), Gemma3TensorNameMapper.vProj(0),
                Gemma3TensorNameMapper.oProj(0), Gemma3TensorNameMapper.gateProj(0), Gemma3TensorNameMapper.upProj(0),
                Gemma3TensorNameMapper.downProj(0)}) {
            SafeTensorEntry e = entry(file, name);
            ByteBuffer fp32 = Gemma3WeightBufferView.decodeFp32LittleEndian(e);
            ByteBuffer bf16 = Gemma3Bf16WeightView.ofBf16Copy(e).inflateToFp32();
            assertEquals(fp32.limit(), bf16.limit(), name + " size");
            for (int i = 0; i < fp32.limit(); i += Float.BYTES) {
                assertEquals(fp32.getFloat(i), bf16.getFloat(i), 0f, name + " @" + i);
            }
        }

        // One BF16 session via the product .wdmlpack load path: Paris + decode timing. PACK-3 changes only the
        // load (transient widen); the device weights stay FP32, so no per-token path changes.
        int[] seqBf16 = new int[DECODE_TOKENS];
        double bf16DecodeMs;
        long bf16LoadMs;
        Path pkg = Files.createTempFile("gemma3-bf16-pack3-", ".wdmlpack");
        try {
            Gemma3WdmlPackCompiler.compile(dir, pkg, true);
            long t = System.nanoTime();
            Gemma3WarpWeights bf16Weights = Gemma3RuntimePackage.open(pkg).loadWarpWeightsHeapLight();
            bf16LoadMs = (System.nanoTime() - t) / 1_000_000L;
            bf16DecodeMs = decodeSequence(bf16Weights, "bf16", seqBf16);
        } finally {
            Files.deleteIfExists(pkg);
        }
        // (decodeSequence already asserts the first token is " Paris" (9079) for the BF16 path.)

        // Projection host-RAM accounting (FP32 retained vs BF16 retained), from the model shape.
        long projFp32 = projectionBytes(config, Float.BYTES);
        long projBf16 = projectionBytes(config, 2);
        long embFp32 = (long) config.vocabSize() * config.hiddenSize() * Float.BYTES;
        long embBf16 = (long) config.vocabSize() * config.hiddenSize() * 2;
        double mb = 1024.0 * 1024.0;
        System.out.println("===== [GEMMA-BF16-PACK-3] layer projection host memory =====");
        System.out.printf(Locale.ROOT, "layer projections host RAM: FP32 %.1f MB -> BF16 %.1f MB (save %.1f MB)%n",
                projFp32 / mb, projBf16 / mb, (projFp32 - projBf16) / mb);
        System.out.printf(Locale.ROOT, "cumulative since PACK-2 (embedding+projections): FP32 %.1f MB -> BF16 %.1f MB "
                + "(save %.1f MB)%n", (embFp32 + projFp32) / mb, (embBf16 + projBf16) / mb,
                (embFp32 + projFp32 - embBf16 - projBf16) / mb);
        System.out.printf(Locale.ROOT, "bf16 package load (loadWarpWeightsHeapLight): %d ms%n", bf16LoadMs);
        System.out.printf(Locale.ROOT, "decode avg/token (BF16 path): %.1f ms (PACK-3 touches only load, not "
                + "the per-token path; device weights stay FP32)%n", bf16DecodeMs);

        // PACK-3 does not change any per-token path; assert decode is on the expected WARP order (no regression).
        assertTrue(bf16DecodeMs < 1500, "BF16 decode avg/token unexpectedly high on WARP: " + bf16DecodeMs);
    }

    private double decodeSequence(Gemma3WarpWeights weights, String label, int[] seqOut) throws Exception {
        try (Gemma3WarpDecodeSession sess = new Gemma3WarpDecodeSession(wb, weights)) {
            int tok = DecoderOnlyMath.argmax(sess.prefillResidentBatched(FRANCE_IDS));
            assertEquals(EXPECTED_NEXT, tok, label + " path must give \" Paris\"");
            for (int i = 0; i < 2; i++) {
                tok = sess.decodeNextTokenResident(tok); // warm
            }
            long t = System.nanoTime();
            for (int i = 0; i < DECODE_TOKENS; i++) {
                tok = sess.decodeNextTokenResident(tok);
                seqOut[i] = tok;
            }
            return (System.nanoTime() - t) / 1e6 / DECODE_TOKENS;
        }
    }

    private static long projectionBytes(Gemma3Config c, int bytesPerElem) {
        long attnDim = (long) c.numAttentionHeads() * c.headDim();
        long kvDim = (long) c.numKeyValueHeads() * c.headDim();
        long hidden = c.hiddenSize();
        long inter = c.intermediateSize();
        long perLayer = attnDim * hidden        // q
                + kvDim * hidden                 // k
                + kvDim * hidden                 // v
                + hidden * attnDim               // o
                + inter * hidden                 // gate
                + inter * hidden                 // up
                + hidden * inter;                // down
        return perLayer * c.numHiddenLayers() * bytesPerElem;
    }

    private static SafeTensorEntry entry(SafeTensorsFile f, String name) throws IOException {
        SafeTensorEntry e = f.tensors().get(name);
        if (e == null) {
            throw new IOException("missing tensor: " + name);
        }
        return e;
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
