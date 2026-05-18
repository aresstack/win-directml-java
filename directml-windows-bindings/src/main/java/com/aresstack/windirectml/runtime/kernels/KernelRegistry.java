package com.aresstack.windirectml.runtime.kernels;

/**
 * Bündelt die für ein Modell-Profil zusammengehörigen Kernel-Implementierungen.
 * <p>
 * Architektur-Adapter (z. B. {@code MiniLmArchitecture},
 * {@code Phi3Architecture}) wählen die passenden Kernel über diese
 * Registry, ohne direkt gegen DirectML-Klassen zu programmieren.
 *
 * <p><b>TODO</b> (modellfamilien-neutral, in dieser Reihenfolge):
 * <ul>
 *   <li>{@link LinearKernel} – Encoder-Pflicht</li>
 *   <li>{@link LayerNormKernel} – Encoder-Pflicht</li>
 *   <li>{@link GeluKernel} – Encoder-Pflicht</li>
 *   <li>{@link AttentionKernel} – Encoder-Pflicht</li>
 *   <li>{@link MeanPoolingKernel} – Encoder-Pflicht</li>
 *   <li>{@link L2NormalizeKernel} – Encoder-Pflicht</li>
 *   <li>{@link QuantizedLinearKernel} – Decoder (Phi-3)</li>
 *   <li>{@link RmsNormKernel} – Decoder (Phi-3, Llama)</li>
 *   <li>{@link SwiGluKernel} – Decoder (Phi-3, Llama)</li>
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

