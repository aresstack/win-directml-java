package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.windows.WindowsBindings;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.function.IntConsumer;

/**
 * Native Java/WARP Gemma 3 workbench runtime (GEMMA-WARP-11): the experimental path that replaces the
 * external Python/Transformers probe when {@code -Dgemma.runtime=native-warp} is set.
 *
 * <p>Pipeline: {@link Gemma3Tokenizer} (+ optional {@link Gemma3ChatTemplate}) → weights from the
 * compiled {@code .wdmlpack} via {@link Gemma3RuntimePackage} → {@link Gemma3WarpGenerator} →
 * detokenized text. It loads <b>only</b> from the package (not raw HF files) for weights, and the
 * tokenizer from {@code tokenizer.json}. An explicit native run either succeeds or fails with a clear
 * message — it never falls back to Python. Requires a DirectML/D3D12 device
 * ({@link WindowsBindings#isSupported()}).</p>
 *
 * <p>Weights load heap-light (GEMMA-WARP-13a): {@link Gemma3RuntimePackage#loadWarpWeightsHeapLight}
 * decodes the layer projections and the tied embedding/LM head into direct FP32 ByteBuffers (off-heap),
 * so the JVM heap no longer carries the ~1 GB {@code float[]} reference weights. Perf work (fused
 * pipeline, batched prefill) remains separate later slices.</p>
 */
public final class Gemma3NativeWarpRuntime {

    private final Path packagePath;
    private final Path tokenizerJson;

    public Gemma3NativeWarpRuntime(Path packagePath, Path tokenizerJson) {
        this.packagePath = Objects.requireNonNull(packagePath, "packagePath");
        this.tokenizerJson = Objects.requireNonNull(tokenizerJson, "tokenizerJson");
    }

    /** Result of a native WARP generation. */
    public record Result(String text, int promptTokens, int outputTokens,
                         Gemma3GenerationResult.FinishReason finishReason, Path packagePath, String backend) {
    }

    /**
     * Resolve the default Gemma native package path inside a model directory
     * ({@code model_gemma3.wdmlpack}).
     */
    public static Path defaultPackagePath(Path modelDir) {
        return modelDir.resolve(Gemma3WdmlPackCompiler.DEFAULT_OUTPUT_NAME);
    }

    /**
     * The actionable message when the native package is missing, or {@code null} when present. Callers
     * that set {@code -Dgemma.runtime=native-warp} must surface this rather than running Python.
     */
    public static String describeMissingPackage(Path packagePath) {
        if (packagePath == null || !Files.isRegularFile(packagePath)) {
            return "Gemma native WARP requires a compiled .wdmlpack package (" + packagePath + "). "
                    + "Use Download/Convert first or run the Gemma compiler.";
        }
        return null;
    }

    /**
     * Generate from an already-rendered prompt string. {@code applyChatTemplate} wraps the text in the
     * Gemma single-user-turn template; pass {@code false} for a raw completion.
     */
    public Result generate(String promptText, boolean applyChatTemplate, int maxNewTokens, IntConsumer onToken)
            throws IOException {
        Objects.requireNonNull(promptText, "promptText");
        if (maxNewTokens < 1) {
            throw new IllegalArgumentException("maxNewTokens must be positive");
        }
        if (!WindowsBindings.isSupported()) {
            throw new IllegalStateException("Gemma native WARP requires a DirectML/D3D12 device, "
                    + "which is not available on this host. Use -Dgemma.runtime=external instead.");
        }
        String missing = describeMissingPackage(packagePath);
        if (missing != null) {
            throw new IllegalStateException(missing);
        }
        if (!Files.isRegularFile(tokenizerJson)) {
            throw new IllegalStateException("Gemma native WARP requires tokenizer.json: " + tokenizerJson);
        }

        Gemma3RuntimePackage pkg = Gemma3RuntimePackage.open(packagePath);
        Gemma3Config config = pkg.config();
        // Heap-light: projections + tied embedding/LM head load as direct FP32 ByteBuffers (off-heap),
        // not the ~1 GB float[] reference path. The reference path stays for parity/tests only.
        Gemma3WarpWeights weights = pkg.loadWarpWeightsHeapLight();

        Gemma3Tokenizer tokenizer = Gemma3Tokenizer.load(tokenizerJson);
        String rendered = applyChatTemplate ? Gemma3ChatTemplate.renderUserTurn(promptText) : promptText;
        int[] promptIds = tokenizer.encode(rendered, true);

        Gemma3StopTokenPolicy stop = stopPolicy(config, tokenizer);

        try {
            WindowsBindings wb = new WindowsBindings();
            wb.init("directml");
            try (Gemma3WarpDecodeSession session = new Gemma3WarpDecodeSession(wb, weights)) {
                Gemma3WarpGenerator generator = new Gemma3WarpGenerator(session, stop);
                Gemma3GenerationResult result = generator.generate(
                        new Gemma3GenerationRequest(promptIds, maxNewTokens), onToken);
                String text = tokenizer.decode(result.generatedTokenIds());
                return new Result(text, result.promptTokenCount(), result.outputTokenCount(),
                        result.finishReason(), packagePath, "WARP");
            } finally {
                wb.close();
            }
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IOException("Gemma native WARP generation failed", e);
        }
    }

    private static Gemma3StopTokenPolicy stopPolicy(Gemma3Config config, Gemma3Tokenizer tokenizer) {
        try {
            int[] eot = tokenizer.encode(Gemma3ChatTemplate.END_OF_TURN, false);
            if (eot.length == 1) {
                return Gemma3StopTokenPolicy.ofEosAndEndOfTurn(config, eot[0]);
            }
        } catch (RuntimeException ignored) {
            // fall back to eos-only if the token can't be resolved
        }
        return Gemma3StopTokenPolicy.ofEos(config);
    }
}
