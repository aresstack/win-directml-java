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
 *   <li>{@link GeluKernel} – ✅ Strategie-Interface mit Feature-Level-Switch
 *       via {@link GeluKernel#create(com.aresstack.windirectml.runtime.DirectMlContextImpl, int)}:
 *       nativer {@link DirectMlGeluKernel}
 *       ({@code DML_OPERATOR_ACTIVATION_GELU}, Enum-ID 157, FL 5.1) auf
 *       neueren DLLs, sonst Composite-Fallback
 *       {@link DirectMlCompositeGeluKernel}
 *       (ERF + IDENTITY + MULTIPLY, alle FL 2.0). Damit läuft GELU auch
 *       auf Windows-11-In-Box {@code DirectML.dll} 1.8.0 (FL 5.0) ohne
 *       Redist. GPU-getestet auf beiden Pfaden.</li>
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
 *   <li>{@link MeanPoolingKernel} – ✅ {@link DirectMlMeanPoolKernel}
 *       ({@code DML_OPERATOR_GEMM} FL 1.0, vorab CPU-normalisierte Gewichte
 *       {@code w[t]=m[t]/Σm}). GPU-getestet, läuft auf jeder ausgelieferten
 *       {@code DirectML.dll}.</li>
 *   <li>{@link L2NormalizeKernel} – ✅ {@link DirectMlL2NormalizeKernel}
 *       (GEMM sum-of-squares {@code A[1,N] · B[N,1] → s[1,1]} +
 *       {@code DML_OPERATOR_ELEMENT_WISE_SQRT} mit ε² in
 *       {@code DML_SCALE_BIAS} + {@code DML_OPERATOR_ELEMENT_WISE_DIVIDE}
 *       mit Broadcast über Nullstrides). Alle drei Sub-Ops sind FL 1.0 –
 *       läuft auf jeder ausgelieferten {@code DirectML.dll}, einschließlich
 *       Windows-11-In-Box 1.8.0. GPU-getestet; der finale 384-Float-Vektor
 *       bleibt bis zum letzten Download komplett auf der GPU.</li>
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

