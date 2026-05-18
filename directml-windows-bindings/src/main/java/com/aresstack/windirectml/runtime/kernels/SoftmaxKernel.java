package com.aresstack.windirectml.runtime.kernels;

import com.aresstack.windirectml.runtime.DirectMlRuntimeException;
import com.aresstack.windirectml.runtime.DirectMlTensor;

/**
 * Numerisch stabile Softmax über die innerste Tensor-Achse:
 * {@code y_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))}.
 * <p>
 * Wird vom Attention-Kernel auf den Score-Tensor angewendet und steht
 * für andere Kernels (Reranker-Logit-Heads etc.) zur Verfügung.
 */
public interface SoftmaxKernel {

    void dispatch(DirectMlTensor x, DirectMlTensor y) throws DirectMlRuntimeException;
}

