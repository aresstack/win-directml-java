package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyMath;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-WARP-12: a measurement probe (no optimization) that records real native-warp vs
 * external-python-transformers numbers for the same prompts. Gated on a DirectML device + the local
 * model; the external half additionally needs a Python with torch + transformers. The printed lines are
 * transcribed into {@code docs/gemma3-warp-runtime-performance.md}.
 */
@EnabledOnOs(OS.WINDOWS)
class Gemma3WarpPerformanceProbeTest {

    private static final int MAX_NEW = 8;
    private static final String[] PROMPTS = {
            "The capital of France is",
            "Fasse in einem Satz zusammen: Die Katze sitzt auf der Matte und schläft.",
            "Erkläre kurz, was READ #EMPLOYEES BY NAME in Natural/ADABAS tut.",
            "Extract JSON {name, city} from: Anna lives in Berlin. Output JSON:"
    };

    @Test
    void probeNativeVsExternal() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML device");
        Path dir = resolveModelDir();
        assumeTrue(dir != null, "Real Gemma 3 270M model not present");

        System.out.println("[PERF] ==== GEMMA-WARP-12 probe (maxNewTokens=" + MAX_NEW + ") ====");
        System.out.println("[PERF] modelDir=" + dir);
        System.out.println("[PERF] java=" + System.getProperty("java.version")
                + " os=" + System.getProperty("os.name") + " maxHeapMB=" + (Runtime.getRuntime().maxMemory() >> 20));

        probeNative(dir);
        probeExternal(dir);
    }

    private void probeNative(Path dir) throws Exception {
        Path pkg = Files.createTempFile("gemma3-perf-", ".wdmlpack");
        try {
            long t = System.nanoTime();
            Gemma3WdmlPackCompiler.compile(dir, pkg, true);
            long compileMs = ms(t);

            t = System.nanoTime();
            Gemma3RuntimePackage rp = Gemma3RuntimePackage.open(pkg);
            Gemma3ReferenceWeights ref = rp.loadReferenceWeights();
            Gemma3WarpWeights weights = Gemma3WarpWeights.from(ref);
            long weightsLoadMs = ms(t);

            Gemma3Tokenizer tokenizer = Gemma3Tokenizer.load(dir.resolve("tokenizer.json"));

            System.gc();
            long heapAfterLoadMB = usedHeapMB();

            WindowsBindings wb = new WindowsBindings();
            wb.init("directml");
            try {
                t = System.nanoTime();
                Gemma3WarpDecodeSession session = new Gemma3WarpDecodeSession(wb, weights);
                long sessionBuildMs = ms(t);
                System.out.println("[PERF native] load: compileMs=" + compileMs + " weightsLoadMs=" + weightsLoadMs
                        + " sessionBuildMs=" + sessionBuildMs + " heapUsedMB=" + heapAfterLoadMB
                        + " (lm-head built lazily on first prefill)");
                try {
                    for (int p = 0; p < PROMPTS.length; p++) {
                        int[] ids = tokenizer.encode(PROMPTS[p], true);
                        long pt = System.nanoTime();
                        float[] logits = session.prefill(ids);
                        long prefillMs = ms(pt);

                        long dt = System.nanoTime();
                        int produced = 0;
                        for (int s = 0; s < MAX_NEW; s++) {
                            int next = DecoderOnlyMath.argmax(logits);
                            produced++;
                            logits = session.decodeNext(next);
                        }
                        long decodeMs = ms(dt);
                        double decodeTokPerSec = produced / (decodeMs / 1000.0);
                        long totalMs = prefillMs + decodeMs;
                        System.out.println("[PERF native] prompt#" + (p + 1)
                                + " promptTokens=" + ids.length + " prefillMs=" + prefillMs
                                + " decodeMs=" + decodeMs + " outputTokens=" + produced
                                + " totalMs=" + totalMs + String.format(" decodeTokPerSec=%.2f", decodeTokPerSec));
                        assertTrue(produced >= 1);
                    }
                } finally {
                    session.close();
                }
            } finally {
                wb.close();
            }
        } finally {
            Files.deleteIfExists(pkg);
        }
    }

    private void probeExternal(Path dir) throws Exception {
        String python = firstWorkingPython();
        if (python == null) {
            System.out.println("[PERF external] SKIPPED: no python with torch+transformers found on PATH");
            return;
        }
        Path script = Files.createTempFile("gemma3-perf-ext-", ".py");
        Files.writeString(script, EXTERNAL_SCRIPT, StandardCharsets.UTF_8);
        try {
            for (int p = 0; p < PROMPTS.length; p++) {
                String line = runPython(python, script, dir, PROMPTS[p]);
                System.out.println("[PERF external] prompt#" + (p + 1) + " " + (line == null ? "FAILED" : line));
            }
        } finally {
            Files.deleteIfExists(script);
        }
    }

    private static String runPython(String python, Path script, Path dir, String prompt) {
        try {
            ProcessBuilder pb = new ProcessBuilder(python, script.toString(), dir.toString(), prompt,
                    String.valueOf(MAX_NEW));
            pb.environment().put("HF_HUB_OFFLINE", "1");
            pb.environment().put("PYTHONIOENCODING", "utf-8");
            pb.redirectErrorStream(false);
            Process proc = pb.start();
            String out = new String(proc.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            String err = new String(proc.getErrorStream().readAllBytes(), StandardCharsets.UTF_8);
            if (!proc.waitFor(600, TimeUnit.SECONDS)) {
                proc.destroyForcibly();
                return "TIMEOUT";
            }
            if (proc.exitValue() != 0) {
                return "exit=" + proc.exitValue() + " err=" + firstLine(err);
            }
            for (String l : out.split("\\R")) {
                if (l.startsWith("EXT ")) {
                    return l.substring(4);
                }
            }
            return "no EXT line; out=" + firstLine(out);
        } catch (Exception e) {
            return "error=" + e.getMessage();
        }
    }

    private static String firstWorkingPython() {
        for (String cand : List.of(System.getProperty("gemma3.python", ""), "python", "python3")) {
            if (cand == null || cand.isBlank()) {
                continue;
            }
            try {
                Process p = new ProcessBuilder(cand, "-c", "import torch, transformers").start();
                if (p.waitFor(120, TimeUnit.SECONDS) && p.exitValue() == 0) {
                    return cand;
                }
            } catch (Exception ignored) {
                // try next candidate
            }
        }
        return null;
    }

    private static final String EXTERNAL_SCRIPT = String.join("\n",
            "import sys, time, torch",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "model_dir, prompt, max_new = sys.argv[1], sys.argv[2], int(sys.argv[3])",
            "t0 = time.time()",
            "tok = AutoTokenizer.from_pretrained(model_dir)",
            "model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)",
            "model.eval()",
            "load_ms = (time.time() - t0) * 1000.0",
            "ids = tok(prompt, return_tensors='pt').input_ids",
            "prompt_tokens = ids.shape[1]",
            "t1 = time.time()",
            "with torch.no_grad():",
            "    out = model.generate(ids, max_new_tokens=max_new, do_sample=False)",
            "gen_ms = (time.time() - t1) * 1000.0",
            "output_tokens = int(out.shape[1] - prompt_tokens)",
            "tps = output_tokens / (gen_ms / 1000.0) if gen_ms > 0 else 0.0",
            "print('EXT load_ms=%.1f gen_ms=%.1f prompt_tokens=%d output_tokens=%d gen_tok_per_sec=%.2f'"
                    + " % (load_ms, gen_ms, prompt_tokens, output_tokens, tps))");

    private static long ms(long startNanos) {
        return (System.nanoTime() - startNanos) / 1_000_000L;
    }

    private static long usedHeapMB() {
        Runtime r = Runtime.getRuntime();
        return (r.totalMemory() - r.freeMemory()) >> 20;
    }

    private static String firstLine(String s) {
        if (s == null || s.isBlank()) {
            return "(none)";
        }
        return s.strip().split("\\R")[0];
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
}
