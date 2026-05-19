package com.aresstack.windirectml.encoder.minilm;

import com.aresstack.windirectml.encoder.PoolingStrategy;
import com.aresstack.windirectml.encoder.bert.BertEncoderConfig;

/**
 * Adapter that bridges the MiniLM-specific {@link MiniLmConfig} to the
 * model-agnostic {@link BertEncoderConfig} the shared DirectML encoder
 * pipeline consumes. MiniLM's pooling/normalisation defaults (mean +
 * L2) are encoded here so MiniLM-callers do not have to know about
 * BERT-family invariants.
 */
final class MiniLmConfigs {

    private MiniLmConfigs() {}

    static BertEncoderConfig toBertConfig(MiniLmConfig c) {
        return new BertEncoderConfig(
                MiniLmArchitecture.NAME,
                c.hiddenSize(),
                c.numLayers(),
                c.numAttentionHeads(),
                c.intermediateSize(),
                c.maxPositionEmbeddings(),
                c.typeVocabSize(),
                c.vocabSize(),
                c.layerNormEps(),
                c.hiddenAct(),
                c.outputDimension(),
                PoolingStrategy.MEAN,
                /* normalize */ true);
    }
}

