package com.aresstack.windirectml.runtime.kernels;

import com.aresstack.windirectml.runtime.DirectMlTensor;
import com.aresstack.windirectml.runtime.DirectMlRuntimeException;

/**
 * GELU-Aktivierung (elementweise). Verwendet von BERT, MiniLM, JinaBERT.
 */
public interface GeluKernel {

    void dispatch(DirectMlTensor x, DirectMlTensor y) throws DirectMlRuntimeException;
}

