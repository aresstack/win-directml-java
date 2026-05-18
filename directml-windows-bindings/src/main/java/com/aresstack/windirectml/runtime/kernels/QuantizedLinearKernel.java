package com.aresstack.windirectml.runtime.kernels;

import com.aresstack.windirectml.runtime.DirectMlTensor;
import com.aresstack.windirectml.runtime.DirectMlRuntimeException;

/**
 * INT4-AWQ-quantisierte Matrixmultiplikation (block-quantisiert).
 * <p>
 * Heute von Phi-3 (AWQ-Block-128) genutzt, später wiederverwendet für
 * andere INT4-Quantisierungen. Skalen- und Zero-Point-Tensoren werden
 * separat übergeben, damit das Layout pro Modellfamilie konfigurierbar
 * bleibt.
 */
public interface QuantizedLinearKernel {

    /**
     * @param x         {@code [M, K]} Aktivierungen (FP16/FP32)
     * @param weight    gepackte 4-Bit-Gewichte
     * @param scales    Skalierungsfaktoren pro Block
     * @param zeros     Zero-Points pro Block (optional, {@code null} für symmetrisch)
     * @param bias      optionaler Bias-Vektor
     * @param y         {@code [M, N]} Ausgabetensor
     * @param blockSize Quantisierungs-Blockgröße (typisch 128)
     */
    void dispatch(DirectMlTensor x,
                  DirectMlTensor weight,
                  DirectMlTensor scales,
                  DirectMlTensor zeros,
                  DirectMlTensor bias,
                  DirectMlTensor y,
                  int blockSize) throws DirectMlRuntimeException;
}

