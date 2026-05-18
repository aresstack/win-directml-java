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
 *       ({@code DML_OPERATOR_ACTIVATION_GELU}, Enum-ID 157, FL 5.1),
 *       benötigt {@code DirectML.dll} ≥ 1.10 – auf älteren in-box
 *       DLLs (1.8.0, Windows 11 RTM) per
 *       {@code -Dwindirectml.directml.dll=...} ein neueres
 *       Microsoft.AI.DirectML-Redistributable einhängen.
 *       GPU-getestet</li>
 *   <li>{@link SoftmaxKernel} – ✅ {@link DirectMlSoftmaxKernel}
 *       ({@code DML_OPERATOR_ACTIVATION_SOFTMAX}, Enum-ID 48, FL 2.0),
 *       läuft auf jeder ausgelieferten {@code DirectML.dll}. GPU-getestet</li>
 *   <li>{@link AttentionKernel} – ✅ {@link DirectMlAttentionKernel}
 *       (Composite-SDPA aus {@code DML_OPERATOR_GEMM} ×2,
 *       optional {@code DML_OPERATOR_ELEMENT_WISE_ADD} (Mask) und
 *       {@code DML_OPERATOR_ACTIVATION_SOFTMAX}). Alle vier
 *       Sub-Ops sind FL 2.0 – läuft auf jeder ausgelieferten
 *       {@code DirectML.dll}, einschließlich Windows-11-RTM
 *       In-Box 1.8.0. GPU-getestet (mit und ohne Mask).</li>
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

    /**
     * Row-wise Softmax über die innerste Tensor-Achse. Baustein für
     * {@link AttentionKernel} sowie für Reranker-Logit-Heads.
     */
    SoftmaxKernel softmax();

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

