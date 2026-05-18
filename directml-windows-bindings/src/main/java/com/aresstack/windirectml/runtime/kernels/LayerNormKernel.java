package com.aresstack.windirectml.runtime.kernels;

import com.aresstack.windirectml.runtime.DirectMlTensor;
import com.aresstack.windirectml.runtime.DirectMlRuntimeException;

/**
 * Standard-LayerNorm (BERT/MiniLM-Stil): {@code y = (x - μ) / √(σ² + ε) · γ + β}.
 * Normalisiert über die letzte Dimension.
 */
public interface LayerNormKernel {

    void dispatch(DirectMlTensor x, DirectMlTensor gamma, DirectMlTensor beta,
                  DirectMlTensor y, float epsilon) throws DirectMlRuntimeException;
}

