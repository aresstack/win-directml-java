package com.aresstack.windirectml.runtime.kernels;

import com.aresstack.windirectml.runtime.DirectMlTensor;
import com.aresstack.windirectml.runtime.DirectMlRuntimeException;

/**
 * SwiGLU-Aktivierung (Llama/Phi-3): {@code y = SiLU(x_gate) ⊙ x_up}.
 * Gate- und Up-Projektion sind bereits getrennte Tensoren.
 */
public interface SwiGluKernel {

    void dispatch(DirectMlTensor xGate, DirectMlTensor xUp, DirectMlTensor y)
            throws DirectMlRuntimeException;
}

