package com.aresstack.windirectml.runtime.kernels;

import com.aresstack.windirectml.runtime.DirectMlTensor;
import com.aresstack.windirectml.runtime.DirectMlRuntimeException;

/**
 * RMSNorm (Llama/Phi-3-Stil): {@code y = x / √(mean(x²) + ε) · γ}.
 * Kein Bias, kein Mittelwert-Abzug.
 */
public interface RmsNormKernel {

    void dispatch(DirectMlTensor x, DirectMlTensor gamma, DirectMlTensor y, float epsilon)
            throws DirectMlRuntimeException;
}

