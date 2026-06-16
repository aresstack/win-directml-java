package com.aresstack.windirectml.inference.gemma;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GEMMA-WORKBENCH-PROFILING-1: the native WARP profile report. The detailed block ("Show runtime profile"
 * on) carries every spec field; the summary block (toggle off) stays short.
 */
class Gemma3NativeWarpProfileReportTest {

    private static Gemma3NativeWarpProfile sampleProfile() {
        return new Gemma3NativeWarpProfile(
                /*packageOpenMs*/ 5, /*tokenizerLoadMs*/ 3, /*weightLoadMs*/ 400, /*sessionInitMs*/ 200,
                /*tokenizeMs*/ 2, /*prefillMs*/ 120, /*decodeTotalMs*/ 600, /*detokenizeMs*/ 1,
                /*runtimeTotalMs*/ 1331, /*promptTokens*/ 28, /*outputTokens*/ 27,
                /*submits*/ 12000, /*fenceWaits*/ 2500, /*readbacks*/ 999,
                /*decodeSubmits*/ 11050, /*decodeFenceWaits*/ 2418, /*decodeReadbacks*/ 962,
                Gemma3WarpExecutionMode.RESIDENT);
    }

    @Test
    void detailedReportCarriesEverySpecField() {
        List<String> lines = Gemma3NativeWarpProfileReport.detailed(
                sampleProfile(), "native-warp-experimental", "WARP", "streaming",
                "model_gemma3.wdmlpack", "tokenizer.json", "summarize", 142, 4, 1400);
        String text = String.join("\n", lines);

        assertTrue(text.contains("Gemma native WARP profile:"), text);
        assertTrue(text.contains("runtime mode: native-warp-experimental"), text);
        assertTrue(text.contains("backend: WARP"), text);
        assertTrue(text.contains("execution: resident-batched"), text);
        assertTrue(text.contains("output mode: streaming"), text);
        assertTrue(text.contains("package: model_gemma3.wdmlpack"), text);
        assertTrue(text.contains("tokenizer: tokenizer.json"), text);
        assertTrue(text.contains("prompt template: summarize"), text);
        assertTrue(text.contains("effective prompt chars: 142"), text);
        assertTrue(text.contains("prompt tokens: 28"), text);
        assertTrue(text.contains("output tokens: 27"), text);
        // load group
        assertTrue(text.contains("load:"), text);
        assertTrue(text.contains("package open: 5 ms"), text);
        assertTrue(text.contains("tokenizer load: 3 ms"), text);
        assertTrue(text.contains("weight load: 400 ms"), text);
        assertTrue(text.contains("WARP/session init: 200 ms"), text);
        // generation group
        assertTrue(text.contains("generation:"), text);
        assertTrue(text.contains("tokenize: 2 ms"), text);
        assertTrue(text.contains("prompt template: 4 ms"), text);
        assertTrue(text.contains("prefill: 120 ms"), text);
        assertTrue(text.contains("decode total: 600 ms"), text);
        assertTrue(text.contains("decode avg/token:"), text);
        assertTrue(text.contains("detokenize: 1 ms"), text);
        assertTrue(text.contains("total: 1400 ms"), text);
        // WARP counters group
        assertTrue(text.contains("WARP counters:"), text);
        assertTrue(text.contains("submits: 12000"), text);
        assertTrue(text.contains("fence waits: 2500"), text);
        assertTrue(text.contains("readbacks: 999"), text);
        assertTrue(text.contains("submits/token:"), text);
        assertTrue(text.contains("fence waits/token:"), text);
        assertTrue(text.contains("readbacks/token:"), text);
    }

    @Test
    void summaryReportIsShortAndOmitsPerPhaseDetail() {
        List<String> lines = Gemma3NativeWarpProfileReport.summary(
                "native-warp-experimental", 28, 27, "STOP_TOKEN", 1400);
        String text = String.join("\n", lines);
        assertTrue(lines.size() <= 6, "summary should be short: " + lines);
        assertTrue(text.contains("Model loaded and generated in 1400 ms"), text);
        assertTrue(text.contains("Runtime mode: native-warp-experimental"), text);
        assertTrue(text.contains("Prompt tokens: 28"), text);
        assertTrue(text.contains("Output tokens: 27"), text);
        assertTrue(text.contains("Finish reason: STOP_TOKEN"), text);
        assertFalse(text.contains("WARP counters:"), "summary must omit the per-phase/counter detail");
        assertFalse(text.contains("prefill:"), "summary must omit the per-phase detail");
    }

    @Test
    void perTokenFiguresUseTheDecodeRegionOverDecodeSteps() {
        // outputTokens=27 -> 26 decodeNext steps; decode total 600 ms / 26 ~= 23.08 ms.
        assertEquals(26, sampleProfile().decodeSteps());
        assertEquals(600.0 / 26, sampleProfile().decodeAvgPerTokenMs(), 1e-9);
        // per-token counters use the decode-region counts (steady-state), not the whole-generate totals.
        assertEquals(11050.0 / 26, sampleProfile().submitsPerToken(), 1e-9);
        assertEquals(2418.0 / 26, sampleProfile().fenceWaitsPerToken(), 1e-9);
        assertEquals(962.0 / 26, sampleProfile().readbacksPerToken(), 1e-9);
    }
}
