package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.windows.WarpSubmissionStats;
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

    /** Result of a native WARP generation, including a phase/WARP-counter {@link Gemma3NativeWarpProfile}. */
    public record Result(String text, int promptTokens, int outputTokens,
                         Gemma3GenerationResult.FinishReason finishReason, Path packagePath, String backend,
                         Gemma3NativeWarpProfile profile) {
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
            return "Gemma native WARP requires a compiled " + Gemma3WdmlPackCompiler.DEFAULT_OUTPUT_NAME
                    + " package. Open the Download tab, select google/gemma-3-270m-it, then run Convert. "
                    + "(expected at " + packagePath + ")";
        }
        return null;
    }

    /** Buffered generation (no per-token callback). */
    public Result generate(String promptText, boolean applyChatTemplate, int maxNewTokens) throws IOException {
        return generateCore(promptText, applyChatTemplate, maxNewTokens, null, null);
    }

    /**
     * Generate from an already-rendered prompt string with a per-token <b>id</b> callback.
     * {@code applyChatTemplate} wraps the text in the Gemma single-user-turn template; pass {@code false}
     * for a raw completion. The stop token is excluded from the callback (it is not part of the output).
     */
    public Result generate(String promptText, boolean applyChatTemplate, int maxNewTokens, IntConsumer onToken)
            throws IOException {
        return generateCore(promptText, applyChatTemplate, maxNewTokens, onToken, null);
    }

    /**
     * Streaming generation: {@code onTextDelta} receives the decoded text for each visible token as it is
     * produced (the concatenation of all deltas equals {@link Result#text()}). The stop token is not
     * streamed. Use this for the Workbench live (streaming) output mode.
     */
    public Result generateStreaming(String promptText, boolean applyChatTemplate, int maxNewTokens,
                                    java.util.function.Consumer<String> onTextDelta) throws IOException {
        return generateCore(promptText, applyChatTemplate, maxNewTokens, null,
                Objects.requireNonNull(onTextDelta, "onTextDelta"));
    }

    private Result generateCore(String promptText, boolean applyChatTemplate, int maxNewTokens,
                                IntConsumer onTokenId, java.util.function.Consumer<String> onTextDelta)
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

        long runtimeStart = System.nanoTime();
        long mark = runtimeStart;
        Gemma3RuntimePackage pkg = Gemma3RuntimePackage.open(packagePath);
        long packageOpenMs = sinceMs(mark);
        Gemma3Config config = pkg.config();
        // Heap-light: projections + tied embedding/LM head load as direct FP32 ByteBuffers (off-heap),
        // not the ~1 GB float[] reference path. The reference path stays for parity/tests only.
        mark = System.nanoTime();
        Gemma3WarpWeights weights = pkg.loadWarpWeightsHeapLight();
        long weightLoadMs = sinceMs(mark);

        mark = System.nanoTime();
        Gemma3Tokenizer tokenizer = Gemma3Tokenizer.load(tokenizerJson);
        long tokenizerLoadMs = sinceMs(mark);
        String rendered = applyChatTemplate ? Gemma3ChatTemplate.renderUserTurn(promptText) : promptText;
        mark = System.nanoTime();
        int[] promptIds = tokenizer.encode(rendered, true);
        long tokenizeMs = sinceMs(mark);

        Gemma3StopTokenPolicy stop = stopPolicy(config, tokenizer);
        // Incremental decode for streaming: decode the growing visible-id prefix and emit only the new tail,
        // so the streamed concatenation matches the final tokenizer.decode of all ids exactly.
        java.util.List<Integer> visibleIds = new java.util.ArrayList<>();
        String[] prevText = {""};
        IntConsumer userCallback = (onTokenId == null && onTextDelta == null) ? null : id -> {
            if (onTokenId != null) {
                onTokenId.accept(id);
            }
            if (onTextDelta != null) {
                visibleIds.add(id);
                int[] ids = visibleIds.stream().mapToInt(Integer::intValue).toArray();
                String full = tokenizer.decode(ids);
                int common = commonPrefixLength(prevText[0], full);
                prevText[0] = full;
                onTextDelta.accept(full.substring(common));
            }
        };
        // Profiling wrapper: the first visible token marks the prefill/decode boundary (prefill = time to
        // first token), then delegate to the user callback. Always installed, even in buffered mode.
        long[] firstTokenNanos = {-1L};
        IntConsumer callback = id -> {
            if (firstTokenNanos[0] < 0) {
                firstTokenNanos[0] = System.nanoTime();
            }
            if (userCallback != null) {
                userCallback.accept(id);
            }
        };

        try {
            WindowsBindings wb = new WindowsBindings();
            long sessionStart = System.nanoTime();
            wb.init("directml");
            try (Gemma3WarpDecodeSession session = new Gemma3WarpDecodeSession(wb, weights)) {
                long sessionInitMs = sinceMs(sessionStart);
                Gemma3WarpGenerator generator = new Gemma3WarpGenerator(session, stop);
                // WARP counters for the generate region only (weight upload happened during session init,
                // before this snapshot, so the deltas isolate prefill + decode).
                WarpSubmissionStats.Snapshot warpBefore = WarpSubmissionStats.snapshot();
                long genStart = System.nanoTime();
                Gemma3GenerationResult result = generator.generate(
                        new Gemma3GenerationRequest(promptIds, maxNewTokens), callback);
                long genEnd = System.nanoTime();
                WarpSubmissionStats.Snapshot warpDelta = WarpSubmissionStats.snapshot().minus(warpBefore);

                long prefillMs = firstTokenNanos[0] > 0
                        ? nanosToMs(firstTokenNanos[0] - genStart) : nanosToMs(genEnd - genStart);
                long decodeTotalMs = firstTokenNanos[0] > 0 ? nanosToMs(genEnd - firstTokenNanos[0]) : 0L;

                long detokenizeStart = System.nanoTime();
                String text = tokenizer.decode(result.generatedTokenIds());
                long detokenizeMs = sinceMs(detokenizeStart);

                Gemma3NativeWarpProfile profile = new Gemma3NativeWarpProfile(
                        packageOpenMs, tokenizerLoadMs, weightLoadMs, sessionInitMs,
                        tokenizeMs, prefillMs, decodeTotalMs, detokenizeMs, sinceMs(runtimeStart),
                        result.promptTokenCount(), result.outputTokenCount(),
                        warpDelta.submits(), warpDelta.fenceWaits(), warpDelta.readbacks());
                return new Result(text, result.promptTokenCount(), result.outputTokenCount(),
                        result.finishReason(), packagePath, "WARP", profile);
            } finally {
                wb.close();
            }
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IOException("Gemma native WARP generation failed", e);
        }
    }

    private static long sinceMs(long startNanos) {
        return nanosToMs(System.nanoTime() - startNanos);
    }

    private static long nanosToMs(long nanos) {
        return nanos / 1_000_000L;
    }

    private static int commonPrefixLength(String a, String b) {
        int n = Math.min(a.length(), b.length());
        int i = 0;
        while (i < n && a.charAt(i) == b.charAt(i)) {
            i++;
        }
        return i;
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
