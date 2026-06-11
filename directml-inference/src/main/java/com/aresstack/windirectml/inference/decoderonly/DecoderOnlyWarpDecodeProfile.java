package com.aresstack.windirectml.inference.decoderonly;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Fine-grained timing accumulator for the decoder-only WARP decode path, broken down per sub-operation.
 *
 * <p>Whether profiling is active and which label prefixes the output are injected by the model family (e.g. SmolLM2
 * reads {@code -Dsmollm2.profile.decode} and labels the breakdown {@code "SmolLM2"}). When disabled, normal runs pay
 * no measurement overhead and produce no profiling output.</p>
 *
 * <p>This is a pure measurement aid: it never changes the numerical result.</p>
 */
public final class DecoderOnlyWarpDecodeProfile {

    private final boolean enabled;
    private final String label;

    public long decodeSteps;

    // Per-layer decode sub-operations (aggregated over all layers and decode steps).
    public long embedding;
    public long attentionNorm;
    public long qkvProjection;
    public long qkvSlice;
    public long rope;
    public long kvAppend;
    public long attentionScore;
    public long softmax;
    public long attentionContext;
    public long outputProjection;
    public long attentionResidual;
    public long mlpNorm;
    // Decode path: gate_up → WARP SwiGLU → down run as one GPU-resident submission, measured as a single timer.
    public long mlpPipeline;
    // Individual MLP stage timers (populated only when the stages run separately, e.g. the batched prefill path).
    public long gateUpProjection;
    public long gateUpSlice;
    public long swiglu;
    public long downProjection;
    public long mlpResidual;

    // Separately tracked stages outside the per-layer loop.
    public long lmHead;
    public long tokenSelect;
    public long streamingCallback;

    /**
     * @param enabled whether timings are collected and emitted (typically read once from a family system property)
     * @param label   prefix for the human-readable breakdown header (e.g. the model family name)
     */
    public DecoderOnlyWarpDecodeProfile(boolean enabled, String label) {
        this.enabled = enabled;
        this.label = Objects.requireNonNull(label, "label");
    }

    public boolean enabled() {
        return enabled;
    }

    public void reset() {
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
        mlpPipeline = 0;
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
    public List<String> format() {
        List<String> lines = new ArrayList<>();
        lines.add(label + " WARP decode micro-profile (decode steps: " + decodeSteps + ", ms aggregated):");
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
        lines.add(line("MLP block (gate_up->SwiGLU->down)", mlpPipeline));
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
        return String.format("  %-34s %6d ms", label, nanos / 1_000_000L);
    }
}
