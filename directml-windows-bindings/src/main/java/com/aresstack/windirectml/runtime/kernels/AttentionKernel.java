package com.aresstack.windirectml.runtime.kernels;

import com.aresstack.windirectml.runtime.DirectMlTensor;
import com.aresstack.windirectml.runtime.DirectMlRuntimeException;

/**
 * Multi-Head-Attention (scaled dot-product).
 * <p>
 * Form-Konvention:
 * <ul>
 *   <li>{@code q, k, v}: {@code [batch, heads, seq, headDim]}</li>
 *   <li>{@code mask}: optional, {@code [batch, 1, 1, seq]} (additiv, -inf an Padding-Positionen)</li>
 *   <li>{@code y}: {@code [batch, heads, seq, headDim]}</li>
 * </ul>
 * <p>
 * Phi-3-spezifische Eigenheiten (RoPE, GQA) werden vor dem Aufruf
 * außerhalb des Kernels aufgelöst.
 */
public interface AttentionKernel {

    void dispatch(DirectMlTensor q, DirectMlTensor k, DirectMlTensor v,
                  DirectMlTensor mask, DirectMlTensor y, float scale)
            throws DirectMlRuntimeException;
}

