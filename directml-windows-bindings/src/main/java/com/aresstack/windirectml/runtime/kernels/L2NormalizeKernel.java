package com.aresstack.windirectml.runtime.kernels;

import com.aresstack.windirectml.runtime.DirectMlTensor;
import com.aresstack.windirectml.runtime.DirectMlRuntimeException;

/**
 * L2-Normalisierung über die letzte Dimension:
 * {@code y_i = x_i / max(||x||₂, ε)}.
 */
public interface L2NormalizeKernel {

    void dispatch(DirectMlTensor x, DirectMlTensor y, float epsilon)
            throws DirectMlRuntimeException;
}

