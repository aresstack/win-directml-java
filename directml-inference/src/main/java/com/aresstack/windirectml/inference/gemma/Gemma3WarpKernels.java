package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.windows.WindowsBindings;
import com.aresstack.windirectml.windows.WindowsNativeException;

/**
 * Shared, stateless WARP compute kernels for Gemma 3 (GEMMA-WARP-9), built once and reused across all
 * layers. These kernels carry no per-layer weights — only the projections do — so building one set and
 * sharing it keeps the resident kernel/PSO/command-list count to a handful regardless of layer count,
 * rather than duplicating ~7 kernels per layer. Requires {@link WindowsBindings#isSupported()}.</p>
 */
public final class Gemma3WarpKernels implements AutoCloseable {

    private final Gemma3WarpRmsNormKernel rmsNorm;
    private final Gemma3WarpQkNormKernel qkNorm;
    private final Gemma3WarpRoPEKernel rope;
    private final Gemma3WarpAttentionScoresKernel scores;
    private final Gemma3WarpSoftmaxKernel softmax;
    private final Gemma3WarpAttentionValueKernel value;
    private final Gemma3WarpGeGluKernel geGlu;
    private final Gemma3WarpElementAddKernel elementAdd;
    private boolean closed;

    public Gemma3WarpKernels(WindowsBindings wb) throws WindowsNativeException {
        this.rmsNorm = new Gemma3WarpRmsNormKernel(wb);
        this.qkNorm = new Gemma3WarpQkNormKernel(wb);
        this.rope = new Gemma3WarpRoPEKernel(wb);
        this.scores = new Gemma3WarpAttentionScoresKernel(wb);
        this.softmax = new Gemma3WarpSoftmaxKernel(wb);
        this.value = new Gemma3WarpAttentionValueKernel(wb);
        this.geGlu = new Gemma3WarpGeGluKernel(wb);
        this.elementAdd = new Gemma3WarpElementAddKernel(wb);
    }

    public Gemma3WarpRmsNormKernel rmsNorm() {
        return rmsNorm;
    }

    public Gemma3WarpQkNormKernel qkNorm() {
        return qkNorm;
    }

    public Gemma3WarpRoPEKernel rope() {
        return rope;
    }

    public Gemma3WarpAttentionScoresKernel scores() {
        return scores;
    }

    public Gemma3WarpSoftmaxKernel softmax() {
        return softmax;
    }

    public Gemma3WarpAttentionValueKernel value() {
        return value;
    }

    public Gemma3WarpGeGluKernel geGlu() {
        return geGlu;
    }

    public Gemma3WarpElementAddKernel elementAdd() {
        return elementAdd;
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            rmsNorm.close();
            qkNorm.close();
            rope.close();
            scores.close();
            softmax.close();
            value.close();
            geGlu.close();
            elementAdd.close();
        }
    }
}
