package com.aresstack.windirectml.inference.smollm2;

import java.util.ArrayList;
import java.util.List;

/**
 * Fine-grained timing accumulator for the WARP decode path, broken down per sub-operation.
 *
 * <p>Disabled by default and only active when {@code -Dsmollm2.profile.decode=true} is set, so normal runs are not
 * polluted with profiling output and pay no measurement overhead. Timings are accumulated across all layers and all
 * decode steps of one generation run and emitted once at the end.</p>
 *
 * <p>This is a pure measurement aid: it never changes the numerical result.</p>
 */
final class SmolLM2WarpDecodeProfile {

    /** Master switch — read once at class load from {@code -Dsmollm2.profile.decode}. */
    static final boolean ENABLED = Boolean.getBoolean("smollm2.profile.decode");

    long decodeSteps;

    // Per-layer decode sub-operations (aggregated over all layers and decode steps).
    long embedding;
    long attentionNorm;
    long qkvProjection;
    long qkvSlice;
    long rope;
    long kvAppend;
    long attentionScore;
    long softmax;
    long attentionContext;
    long outputProjection;
    long attentionResidual;
    long mlpNorm;
    long gateUpProjection;
    long gateUpSlice;
    long swiglu;
    long downProjection;
    long mlpResidual;

    // Separately tracked stages outside the per-layer loop.
    long lmHead;
    long tokenSelect;
    long streamingCallback;

    boolean enabled() {
        return ENABLED;
    }

    void reset() {
        decodeSteps = 0;
        embedding = 0;
        attentionNorm = 0;
        qkvProjection = 0;
        qkvSlice = 0;
        rope = 0;
        kvAppend = 0;
        attentionScore = 0;
        softmax = 0;
        attentionContext = 0;
        outputProjection = 0;
        attentionResidual = 0;
        mlpNorm = 0;
        gateUpProjection = 0;
        gateUpSlice = 0;
        swiglu = 0;
        downProjection = 0;
        mlpResidual = 0;
        lmHead = 0;
        tokenSelect = 0;
        streamingCallback = 0;
    }

    /** Human-readable breakdown (one entry per measured sub-operation, in milliseconds). */
    List<String> format() {
        List<String> lines = new ArrayList<>();
        lines.add("SmolLM2 WARP decode micro-profile (decode steps: " + decodeSteps + ", ms aggregated):");
        lines.add(line("embedding", embedding));
        lines.add(line("attention RMSNorm", attentionNorm));
        lines.add(line("qkv projection", qkvProjection));
        lines.add(line("qkv slice/copy", qkvSlice));
        lines.add(line("RoPE", rope));
        lines.add(line("KV-cache append", kvAppend));
        lines.add(line("attention scores", attentionScore));
        lines.add(line("softmax", softmax));
        lines.add(line("attention context", attentionContext));
        lines.add(line("output projection", outputProjection));
        lines.add(line("attention residual", attentionResidual));
        lines.add(line("MLP RMSNorm", mlpNorm));
        lines.add(line("gate-up projection", gateUpProjection));
        lines.add(line("gate-up slice/copy", gateUpSlice));
        lines.add(line("SwiGLU", swiglu));
        lines.add(line("down projection", downProjection));
        lines.add(line("MLP residual", mlpResidual));
        lines.add(line("lm-head projection", lmHead));
        lines.add(line("token selection", tokenSelect));
        lines.add(line("streaming callback", streamingCallback));
        return lines;
    }

    private static String line(String label, long nanos) {
        return String.format("  %-22s %6d ms", label, nanos / 1_000_000L);
    }
}
