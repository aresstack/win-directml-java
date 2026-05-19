package com.aresstack.windirectml.encoder.bert;

import com.aresstack.windirectml.runtime.GpuBuffer;

import java.util.Objects;

/**
 * GPU-resident weights of one BERT-style encoder layer (Q/K/V + attn
 * output, attention LayerNorm, MLP intermediate/output, MLP LayerNorm).
 * <p>
 * Architecturally identical to the per-layer record HuggingFace uses
 * for {@code BertLayer}; this is the generic version the shared
 * {@link DirectMlBertEncoderLayerBlock} consumes. Each buffer is
 * owned by the caller; the layer block only borrows the references.
 */
public record BertGpuLayerWeights(
        GpuBuffer qWeight, GpuBuffer qBias,
        GpuBuffer kWeight, GpuBuffer kBias,
        GpuBuffer vWeight, GpuBuffer vBias,
        GpuBuffer attnOutWeight, GpuBuffer attnOutBias,
        GpuBuffer attnLnGamma, GpuBuffer attnLnBeta,
        GpuBuffer mlpInterWeight, GpuBuffer mlpInterBias,
        GpuBuffer mlpOutWeight, GpuBuffer mlpOutBias,
        GpuBuffer outLnGamma, GpuBuffer outLnBeta
) {
    public BertGpuLayerWeights {
        Objects.requireNonNull(qWeight,        "qWeight");
        Objects.requireNonNull(qBias,          "qBias");
        Objects.requireNonNull(kWeight,        "kWeight");
        Objects.requireNonNull(kBias,          "kBias");
        Objects.requireNonNull(vWeight,        "vWeight");
        Objects.requireNonNull(vBias,          "vBias");
        Objects.requireNonNull(attnOutWeight,  "attnOutWeight");
        Objects.requireNonNull(attnOutBias,    "attnOutBias");
        Objects.requireNonNull(attnLnGamma,    "attnLnGamma");
        Objects.requireNonNull(attnLnBeta,     "attnLnBeta");
        Objects.requireNonNull(mlpInterWeight, "mlpInterWeight");
        Objects.requireNonNull(mlpInterBias,   "mlpInterBias");
        Objects.requireNonNull(mlpOutWeight,   "mlpOutWeight");
        Objects.requireNonNull(mlpOutBias,     "mlpOutBias");
        Objects.requireNonNull(outLnGamma,     "outLnGamma");
        Objects.requireNonNull(outLnBeta,      "outLnBeta");
    }
}

