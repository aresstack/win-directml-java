package com.aresstack.windirectml.runtime.kernels;

import com.aresstack.windirectml.runtime.DirectMlTensor;
import com.aresstack.windirectml.runtime.DirectMlRuntimeException;

/**
 * Mean Pooling über die Sequenz-Dimension, gewichtet mit der Attention-Mask.
 * <p>
 * Form-Konvention:
 * <ul>
 *   <li>{@code tokenEmbeddings}: {@code [batch, seq, hidden]}</li>
 *   <li>{@code attentionMask}:   {@code [batch, seq]} (Werte 0 oder 1)</li>
 *   <li>{@code y}: {@code [batch, hidden]}</li>
 * </ul>
 */
public interface MeanPoolingKernel {

    void dispatch(DirectMlTensor tokenEmbeddings, DirectMlTensor attentionMask, DirectMlTensor y)
            throws DirectMlRuntimeException;
}

