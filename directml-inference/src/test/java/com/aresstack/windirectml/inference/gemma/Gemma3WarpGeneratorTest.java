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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-WARP-10b: full native WARP greedy generation ({@link Gemma3WarpGenerator}) on top of the verified
 * decode session.
 *
 * <ul>
 *   <li><b>Synthetic</b>: multi-step greedy matches a manual greedy loop over the CPU reference; the
 *       max-tokens and stop-token contracts hold; streaming yields exactly the visible output.</li>
 *   <li><b>Real model</b>: "The capital of France is" generates " Paris" as the first token and runs
 *       stably for several steps; streaming == result (no text-quality claim).</li>
 * </ul>
 *
 * <p>Skipped (assumption-aborted) without a DirectML/D3D12 device.</p>
 */
@EnabledOnOs(OS.WINDOWS)
class Gemma3WarpGeneratorTest {

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

    private static Gemma3Config config() {
        return new Gemma3Config("gemma3_text", List.of("Gemma3ForCausalLM"),
                8, 16, 4, 2, 1, 4, 32, 32768, 1e-6, 1_000_000, 10_000, 512, 2, List.of(),
                4, "gelu_pytorch_tanh", 2, 1, 0, true);
    }

    @Test
    void syntheticGreedyMatchesReferenceAndContractIsConsistent() throws Exception {
        Gemma3Config config = config();
        Gemma3ReferenceWeights ref = syntheticWeights(config, new Random(2001));
        int[] prompt = {7, 3, 19, 0, 25};
        int n = 5;
        int[] expected = manualGreedy(new Gemma3ReferenceForwardPass(ref), prompt, n);

        try (Gemma3WarpDecodeSession sess = new Gemma3WarpDecodeSession(wb, Gemma3WarpWeights.from(ref))) {
            // never-stop policy so this is a clean greedy comparison
            Gemma3WarpGenerator gen = new Gemma3WarpGenerator(sess, Gemma3StopTokenPolicy.of());
            Gemma3GenerationResult r = gen.generate(new Gemma3GenerationRequest(prompt, n));

            assertArrayEquals(expected, r.generatedTokenIds(), "greedy ids match the reference");
            assertEquals(Gemma3GenerationResult.FinishReason.MAX_TOKENS, r.finishReason());
            assertEquals(prompt.length, r.promptTokenCount());
            assertEquals(n, r.outputTokenCount());
            assertEquals(n, r.generatedTokenIds().length);
            int[] full = Arrays.copyOf(prompt, prompt.length + n);
            System.arraycopy(expected, 0, full, prompt.length, n);
            assertArrayEquals(full, r.fullTokenIds(), "fullTokenIds = prompt + generated");
        }
    }

    @Test
    void stopTokenEndsGenerationAndIsNotInOutputNorStreamed() throws Exception {
        Gemma3Config config = config();
        Gemma3ReferenceWeights ref = syntheticWeights(config, new Random(2003));
        int[] prompt = {1, 9, 14, 30};

        try (Gemma3WarpDecodeSession sess = new Gemma3WarpDecodeSession(wb, Gemma3WarpWeights.from(ref))) {
            // first find what greedy would produce first, then make THAT the stop token
            int first = new Gemma3WarpGenerator(sess, Gemma3StopTokenPolicy.of())
                    .generate(new Gemma3GenerationRequest(prompt, 1)).generatedTokenIds()[0];

            List<Integer> streamed = new ArrayList<>();
            Gemma3WarpGenerator gen = new Gemma3WarpGenerator(sess, Gemma3StopTokenPolicy.of(first));
            Gemma3GenerationResult r = gen.generate(new Gemma3GenerationRequest(prompt, 5), streamed::add);

            assertEquals(0, r.generatedTokenIds().length, "stop on the first token -> empty visible output");
            assertEquals(Gemma3GenerationResult.FinishReason.STOP_TOKEN, r.finishReason());
            assertArrayEquals(prompt, r.fullTokenIds(), "no tokens added");
            assertTrue(streamed.isEmpty(), "stop token must not be streamed");
        }
    }

    @Test
    void streamingMatchesVisibleResult() throws Exception {
        Gemma3Config config = config();
        Gemma3ReferenceWeights ref = syntheticWeights(config, new Random(2007));
        int[] prompt = {4, 4, 21, 8};

        try (Gemma3WarpDecodeSession sess = new Gemma3WarpDecodeSession(wb, Gemma3WarpWeights.from(ref))) {
            List<Integer> streamed = new ArrayList<>();
            Gemma3WarpGenerator gen = new Gemma3WarpGenerator(sess, Gemma3StopTokenPolicy.of());
            Gemma3GenerationResult r = gen.generate(new Gemma3GenerationRequest(prompt, 4), streamed::add);

            int[] streamedArr = streamed.stream().mapToInt(Integer::intValue).toArray();
            assertArrayEquals(r.generatedTokenIds(), streamedArr, "streamed == visible result");
        }
    }

    @Test
    void realModelGeneratesParisAndStreamsConsistently() throws Exception {
        Path dir = resolveModelDir();
        assumeTrue(dir != null, "Real Gemma 3 270M model not present");
        Gemma3Config config = new Gemma3ConfigReader().read(dir.resolve("config.json"));
        assumeTrue(config.vocabSize() == 262144, "expected the real 270M-it config");
        SafeTensorsFile file = SafeTensorsReader.read(dir.resolve("model.safetensors"));

        ByteBuffer embBb = decodeEmbeddingDirectFp32(file.tensors().get(Gemma3TensorNameMapper.EMBED_TOKENS));
        float[] finalNorm = Gemma3ReferenceWeights.decodeFloats(file.tensors().get(Gemma3TensorNameMapper.FINAL_NORM));
        Gemma3WarpLayerWeights[] layers = loadWarpLayers(file, config);
        Gemma3WarpWeights weights = Gemma3WarpWeights.ofByteBufferEmbedding(config, embBb, finalNorm, layers);

        Gemma3StopTokenPolicy stop = Gemma3StopTokenPolicy.ofEos(config);
        Path tok = dir.resolve("tokenizer.json");
        Gemma3Tokenizer tokenizer = Files.isRegularFile(tok) ? Gemma3Tokenizer.load(tok) : null;
        if (tokenizer != null) {
            int eot = tokenizer.encode("<end_of_turn>", false)[0];
            stop = Gemma3StopTokenPolicy.ofEosAndEndOfTurn(config, eot);
        }

        try (Gemma3WarpDecodeSession sess = new Gemma3WarpDecodeSession(wb, weights)) {
            List<Integer> streamed = new ArrayList<>();
            Gemma3WarpGenerator gen = new Gemma3WarpGenerator(sess, stop);
            Gemma3GenerationResult r = gen.generate(new Gemma3GenerationRequest(FRANCE_IDS, 3), streamed::add);

            assertTrue(r.outputTokenCount() >= 1, "expected at least one generated token");
            assertEquals(EXPECTED_NEXT, r.generatedTokenIds()[0], "first generated token must be \" Paris\"");
            int[] streamedArr = streamed.stream().mapToInt(Integer::intValue).toArray();
            assertArrayEquals(r.generatedTokenIds(), streamedArr, "streamed == visible result");
            assertEquals(FRANCE_IDS.length, r.promptTokenCount());

            String text = tokenizer == null ? "(no tokenizer)" : tokenizer.decode(r.generatedTokenIds());
            System.out.println("[GEMMA-WARP-GEN] ids=" + Arrays.toString(r.generatedTokenIds())
                    + " reason=" + r.finishReason() + " text='" + text + "'");
        }
    }

    // ── helpers ──────────────────────────────────────────────────────

    private static int[] manualGreedy(Gemma3ReferenceForwardPass ref, int[] prompt, int n) {
        int[] seq = Arrays.copyOf(prompt, prompt.length);
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            int next = DecoderOnlyMath.argmax(ref.logitsForLastToken(seq));
            out[i] = next;
            seq = Arrays.copyOf(seq, seq.length + 1);
            seq[seq.length - 1] = next;
        }
        return out;
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
}
