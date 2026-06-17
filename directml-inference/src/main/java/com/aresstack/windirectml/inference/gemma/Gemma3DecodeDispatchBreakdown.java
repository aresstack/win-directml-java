package com.aresstack.windirectml.inference.gemma;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Per-decode-token dispatch breakdown for the native WARP/DirectML Gemma decode path (GEMMA-WARP-15).
 *
 * <p>A small, Swing-free analysis model of the kernels that {@link Gemma3WarpLayer#decodeStepResident}
 * records per layer, plus the per-token tail ({@link Gemma3WarpDecodeSession#residentLogits}: final
 * RMSNorm + tied LM-head matvec). It groups the dispatches by category so the largest groups of small
 * WARP dispatches — the per-dispatch/per-UAV-barrier cost on the CPU rasterizer that dominates decode
 * (see {@code gemma3-warp-performance-ceiling.md}) — can be identified and a fusion candidate chosen.</p>
 *
 * <p>The model is validated against the empirically measured count: {@link #baseline} totals 380
 * dispatches/token for the real 18-layer model (the GEMMA-WARP-14b measurement) and {@link #current}
 * totals 344 after the GEMMA-WARP-15 norm+add fusion. {@code dmlGemm} groups are the large projections
 * that stay on DML-GEMM (not fusion targets); {@code fuseableSmall} are the small element/attention WARP
 * dispatches.</p>
 */
public final class Gemma3DecodeDispatchBreakdown {

    /** One category of dispatches, counted per layer (the tail is modelled separately). */
    public record Group(String name, int perLayer, boolean fuseableSmall, boolean dmlGemm) {
        public Group {
            if (perLayer < 0) {
                throw new IllegalArgumentException("perLayer must be >= 0: " + name);
            }
        }
    }

    private final int numLayers;
    private final List<Group> groups;
    private final int tailDispatches;

    private Gemma3DecodeDispatchBreakdown(int numLayers, List<Group> groups, int tailDispatches) {
        if (numLayers <= 0) {
            throw new IllegalArgumentException("numLayers must be > 0: " + numLayers);
        }
        this.numLayers = numLayers;
        this.groups = List.copyOf(groups);
        this.tailDispatches = tailDispatches;
    }

    /**
     * The pre-GEMMA-WARP-15 decode breakdown: 21 dispatches/layer (7 DML-GEMM matvecs + 14 small WARP
     * dispatches) + a 2-dispatch tail (final RMSNorm + LM-head matvec). For 18 layers this totals 380
     * dispatches/token — the GEMMA-WARP-14b measurement.
     */
    public static Gemma3DecodeDispatchBreakdown baseline(int numLayers) {
        List<Group> g = new ArrayList<>();
        // Large projections — DML-GEMM, kept (not fusion targets).
        g.add(new Group("qkvo-projection (DML-GEMM)", 4, false, true));
        g.add(new Group("mlp gate/up/down (DML-GEMM)", 3, false, true));
        // Small WARP element/attention dispatches.
        g.add(new Group("rmsnorm (input, pre-ff)", 2, true, false));
        g.add(new Group("rmsnorm post-attn/post-ff", 2, true, false));
        g.add(new Group("residual element-add", 2, true, false));
        g.add(new Group("qk-norm (q, k)", 2, true, false));
        g.add(new Group("rope (q, k)", 2, true, false));
        g.add(new Group("kv-append (k, v)", 2, true, false));
        g.add(new Group("fused attention", 1, true, false));
        g.add(new Group("geglu", 1, true, false));
        return new Gemma3DecodeDispatchBreakdown(numLayers, g, 2);
    }

    /**
     * The current decode breakdown (post GEMMA-WARP-15 norm+add fusion AND GEMMA-WARP-16 projection fusion):
     * q/k/v are one fused QKV DML-GEMM (was 3) and gate/up are one fused GateUp DML-GEMM (was 2), and the
     * post-attn/post-ff RMSNorm+residual-add are fused (15). The layer runs 16 dispatches and the per-token
     * total is 290 for 18 layers (380 → 344 by 15 → 290 by 16).
     */
    public static Gemma3DecodeDispatchBreakdown current(int numLayers) {
        List<Group> g = new ArrayList<>();
        g.add(new Group("qkv projection (fused DML-GEMM)", 1, false, true));
        g.add(new Group("o projection (DML-GEMM)", 1, false, true));
        g.add(new Group("gate+up projection (fused DML-GEMM)", 1, false, true));
        g.add(new Group("down projection (DML-GEMM)", 1, false, true));
        g.add(new Group("rmsnorm (input, pre-ff)", 2, true, false));
        g.add(new Group("fused rmsnorm+residual-add (post-attn, post-ff)", 2, true, false));
        g.add(new Group("qk-norm (q, k)", 2, true, false));
        g.add(new Group("rope (q, k)", 2, true, false));
        g.add(new Group("kv-append (k, v)", 2, true, false));
        g.add(new Group("fused attention", 1, true, false));
        g.add(new Group("geglu", 1, true, false));
        return new Gemma3DecodeDispatchBreakdown(numLayers, g, 2);
    }

    public int numLayers() {
        return numLayers;
    }

    public List<Group> groups() {
        return groups;
    }

    /** Total dispatches in one decoded token: all per-layer groups across every layer + the tail. */
    public int dispatchesPerToken() {
        return dispatchesPerLayer() * numLayers + tailDispatches;
    }

    public int dispatchesPerLayer() {
        return groups.stream().mapToInt(Group::perLayer).sum();
    }

    /** The fuseable small-WARP-dispatch groups, largest (per token) first — the fusion candidates. */
    public List<Group> smallDispatchGroupsLargestFirst() {
        List<Group> small = new ArrayList<>(groups.stream().filter(Group::fuseableSmall).toList());
        small.sort(Comparator.comparingInt((Group x) -> x.perLayer() * numLayers).reversed());
        return small;
    }

    /** A human-readable per-layer + per-token table for logging/docs. */
    public String render() {
        StringBuilder sb = new StringBuilder();
        sb.append("Gemma decode dispatch breakdown (").append(numLayers).append(" layers):\n");
        for (Group grp : groups) {
            sb.append(String.format("  %-48s %2d/layer  %4d/token  %s%n",
                    grp.name(), grp.perLayer(), grp.perLayer() * numLayers,
                    grp.dmlGemm() ? "[DML-GEMM, kept]" : (grp.fuseableSmall() ? "[small WARP]" : "")));
        }
        sb.append(String.format("  %-48s %2s         %4d/token%n", "tail (final rmsnorm + lm-head)", "", tailDispatches));
        sb.append(String.format("  %-48s %2d/layer  %4d/token%n", "TOTAL", dispatchesPerLayer(), dispatchesPerToken()));
        return sb.toString();
    }
}
