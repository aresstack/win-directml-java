package com.aresstack.windirectml.runtime.kernels;

/**
 * Bündelt die für ein Modell-Profil zusammengehörigen Kernel-Implementierungen.
 * <p>
 * Architektur-Adapter (z. B. {@code MiniLmArchitecture},
 * {@code Phi3Architecture}) wählen die passenden Kernel über diese
 * Registry, ohne direkt gegen DirectML-Klassen zu programmieren.
 *
 * <p><b>Implementierungsstand (Encoder-Pflicht):</b>
 * <ul>
 *   <li>{@link LinearKernel} – ✅ {@link DirectMlLinearKernel}
 *       ({@code DML_OPERATOR_GEMM}), GPU-getestet</li>
 *   <li>{@link LayerNormKernel} – ✅ {@link DirectMlLayerNormKernel}
 *       ({@code DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION}, MVN0), GPU-getestet</li>
 *   <li>{@link GeluKernel} – ✅ {@link DirectMlGeluKernel}
 *       (komponiert aus FL-2.0-Primitiven ERF + IDENTITY + MULTIPLY,
 *       läuft auf jeder in-box {@code C:\Windows\System32\DirectML.dll}),
 *       GPU-getestet</li>
 *   <li>{@link AttentionKernel} – ⏳ nächster Sprint</li>
 *   <li>{@link MeanPoolingKernel} – ⏳ Encoder-Pflicht</li>
 *   <li>{@link L2NormalizeKernel} – ⏳ Encoder-Pflicht</li>
 * </ul>
 *
 * <p><b>Decoder-Pflicht (Phi-3 / Llama – nicht für Encoder-Pfad benötigt):</b>
 * <ul>
 *   <li>{@link QuantizedLinearKernel} – ⏳</li>
 *   <li>{@link RmsNormKernel} – ⏳</li>
 *   <li>{@link SwiGluKernel} – ⏳</li>
 * </ul>
 */
public interface KernelRegistry {

    LinearKernel linear();

    LayerNormKernel layerNorm();

    GeluKernel gelu();

    AttentionKernel attention();

    MeanPoolingKernel meanPooling();

    L2NormalizeKernel l2Normalize();

    /**
     * Optional: nur Decoder-Pfade benötigen quantisierte MatMul.
     */
    QuantizedLinearKernel quantizedLinear();

    /**
     * Optional: RMSNorm wird vom Encoder-Pfad nicht benötigt.
     */
    RmsNormKernel rmsNorm();

    /**
     * Optional: SwiGLU wird vom Encoder-Pfad nicht benötigt.
     */
    SwiGluKernel swiGlu();
}

