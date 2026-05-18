package com.aresstack.windirectml.runtime.kernels;

import com.aresstack.windirectml.runtime.DirectMlTensor;
import com.aresstack.windirectml.runtime.DirectMlRuntimeException;

/**
 * Dichte Matrixmultiplikation: {@code y = x · W^T + b}.
 * <p>
 * Form-Konvention:
 * <ul>
 *   <li>{@code x}: {@code [M, K]}</li>
 *   <li>{@code W}: {@code [N, K]} (transponiert beim Compute)</li>
 *   <li>{@code b}: {@code [N]}, optional</li>
 *   <li>{@code y}: {@code [M, N]}</li>
 * </ul>
 */
public interface LinearKernel {

    void dispatch(DirectMlTensor x, DirectMlTensor weight, DirectMlTensor bias, DirectMlTensor y)
            throws DirectMlRuntimeException;
}

