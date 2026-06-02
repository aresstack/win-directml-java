package com.aresstack.windirectml.encoder.bert;

import java.util.Objects;

/**
 * CPU-resident weights of one BERT-style encoder layer (16 dense
 * float arrays). Generic counterpart to the GPU-side
 * {@link BertGpuLayerWeights}; consumed by {@link CpuBertEncoder}
 * and uploaded to the device by {@link DirectMlBertEncoder}.
 */
public record BertCpuLayerWeights(
        float[] qWeight, float[] qBias,
        float[] kWeight, float[] kBias,
        float[] vWeight, float[] vBias,
        float[] attnOutWeight, float[] attnOutBias,
        float[] attnLnGamma, float[] attnLnBeta,
        float[] mlpInterWeight, float[] mlpInterBias,
        float[] mlpOutWeight, float[] mlpOutBias,
        float[] outLnGamma, float[] outLnBeta
) {
    public BertCpuLayerWeights {
        Objects.requireNonNull(qWeight);
        Objects.requireNonNull(qBias);
        Objects.requireNonNull(kWeight);
        Objects.requireNonNull(kBias);
        Objects.requireNonNull(vWeight);
        Objects.requireNonNull(vBias);
        Objects.requireNonNull(attnOutWeight);
        Objects.requireNonNull(attnOutBias);
        Objects.requireNonNull(attnLnGamma);
        Objects.requireNonNull(attnLnBeta);
        Objects.requireNonNull(mlpInterWeight);
        Objects.requireNonNull(mlpInterBias);
        Objects.requireNonNull(mlpOutWeight);
        Objects.requireNonNull(mlpOutBias);
        Objects.requireNonNull(outLnGamma);
        Objects.requireNonNull(outLnBeta);
    }
}

