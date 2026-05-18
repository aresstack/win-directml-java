package com.aresstack.windirectml.encoder.minilm;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.EncoderArchitecture;
import com.aresstack.windirectml.encoder.EncoderWeights;
import com.aresstack.windirectml.encoder.PoolingStrategy;

import java.nio.file.Path;

/**
 * Architecture descriptor for {@code sentence-transformers/all-MiniLM-L6-v2}.
 * <p>
 * <b>Reference structure</b> (6-layer BERT-tiny):
 * <ul>
 *   <li>hidden size: 384</li>
 *   <li>layers: 6</li>
 *   <li>attention heads: 12</li>
 *   <li>intermediate size: 1536</li>
 *   <li>max sequence length: 512 (sentence-transformers config uses 256)</li>
 *   <li>output dimension: 384</li>
 *   <li>activation: GELU</li>
 *   <li>normalization: LayerNorm</li>
 *   <li>position embeddings: absolute</li>
 *   <li>pooling: mean over token embeddings, masked by attention</li>
 *   <li>post-pooling: L2-normalize</li>
 * </ul>
 *
 * <b>Expected tensor names (BERT-style)</b>:
 * <pre>
 *   embeddings.word_embeddings.weight              [vocab,  384]
 *   embeddings.position_embeddings.weight          [maxPos, 384]
 *   embeddings.token_type_embeddings.weight        [2,      384]
 *   embeddings.LayerNorm.weight / .bias            [384]
 *   encoder.layer.{i}.attention.self.query.weight  [384, 384] (+ bias)
 *   encoder.layer.{i}.attention.self.key.weight    [384, 384] (+ bias)
 *   encoder.layer.{i}.attention.self.value.weight  [384, 384] (+ bias)
 *   encoder.layer.{i}.attention.output.dense.*     [384, 384] (+ bias)
 *   encoder.layer.{i}.attention.output.LayerNorm.* [384]
 *   encoder.layer.{i}.intermediate.dense.*         [1536, 384] (+ bias)
 *   encoder.layer.{i}.output.dense.*               [384, 1536] (+ bias)
 *   encoder.layer.{i}.output.LayerNorm.*           [384]
 *   pooler.*                                       (unused for sentence-transformers)
 * </pre>
 *
 * <b>Required model artifacts</b>:
 * <ul>
 *   <li>{@code config.json}</li>
 *   <li>{@code tokenizer.json} (WordPiece, uncased)</li>
 *   <li>{@code model.safetensors}</li>
 *   <li>{@code 1_Pooling/config.json} (sentence-transformers pooling config)</li>
 * </ul>
 *
 * <p><b>Status:</b> descriptor only. Implementation lands once the
 * DirectML runtime core is extracted (issues 11–13).
 */
public final class MiniLmArchitecture implements EncoderArchitecture {

    public static final String NAME = "sentence-transformers/all-MiniLM-L6-v2";

    private final MiniLmConfig config;

    public MiniLmArchitecture() {
        this(MiniLmConfig.defaults());
    }

    public MiniLmArchitecture(MiniLmConfig config) {
        this.config = config;
    }

    public MiniLmConfig config() { return config; }

    @Override public String name() { return NAME; }
    @Override public int hiddenSize() { return config.hiddenSize(); }
    @Override public int numLayers() { return config.numLayers(); }
    @Override public int numAttentionHeads() { return config.numAttentionHeads(); }
    @Override public int maxSequenceLength() { return config.maxPositionEmbeddings(); }
    @Override public int outputDimension() { return config.outputDimension(); }
    @Override public PoolingStrategy poolingStrategy() { return PoolingStrategy.MEAN; }
    @Override public boolean l2NormalizeByDefault() { return true; }

    /**
     * Loads MiniLM weights from {@code modelDir} into the CPU reference bundle
     * ({@link CpuMiniLmWeights}). This is the same path used by
     * {@link CpuMiniLmEncoder#load(Path)} and serves as the correctness
     * reference for the upcoming DirectML kernel migration.
     */
    @Override
    public EncoderWeights loadWeights(Path modelDir) throws EmbeddingException {
        return CpuMiniLmWeights.load(modelDir, this);
    }
}
