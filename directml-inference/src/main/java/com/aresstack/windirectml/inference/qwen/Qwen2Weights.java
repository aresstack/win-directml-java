package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.windows.OnnxModelReader;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxGraph;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxNode;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.function.Predicate;

/**
 * Reads Qwen2.5-Coder weights from ONNX model graph + external data file.
 *
 * <p>Qwen models exported via HuggingFace Optimum use the same ONNX-as-weight-container
 * pattern as Phi-3: a {@code model.onnx} file with graph metadata and a
 * {@code model.onnx.data} file with external weight data.
 *
 * <h2>Differences from Phi-3 weight layout</h2>
 * <ul>
 *   <li><b>QKV projections</b>: Qwen uses separate Q, K, V weight matrices with
 *       different output dimensions due to GQA (Q=[hidden,hidden], K/V=[kvSize,hidden]).
 *       Phi-3 has Q=K=V=[hidden,hidden] since num_kv_heads = num_heads.</li>
 *   <li><b>No activation scales</b>: Qwen ONNX models (at least the standard
 *       optimum export) do not have per-projection activation scales that Phi-3 uses.</li>
 *   <li><b>Weight naming</b>: Qwen uses weight names like
 *       {@code model.layers.N.self_attn.q_proj.weight} (external fp16) or graph-traced
 *       MatMulNBits outputs containing {@code q_proj}, {@code k_proj}, etc.</li>
 * </ul>
 *
 * <p>Supports both:
 * <ul>
 *   <li>INT4 quantized weights (MatMulNBits nodes) — same format as Phi-3</li>
 *   <li>FP16 weights (external data tensors) — for non-quantized models</li>
 * </ul>
 */
public final class Qwen2Weights implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(Qwen2Weights.class);

    // ── Public types ─────────────────────────────────────────────────────

    /**
     * INT4 quantized weight block for MatMulNBits.
     * Same format as Phi-3's QuantizedWeight.
     */
    public record QuantizedWeight(
            byte[] qWeight,    // packed uint4 data
            float[] scales,    // per-block scales (fp32)
            byte[] zeroPoints, // packed uint4 zero points
            int N, int K, int blockSize
    ) {
        /** Dequantize and compute y += x @ W^T. */
        public void matvec(float[] x, float[] y) {
            int blocksPerRow = K / blockSize;
            for (int n = 0; n < N; n++) {
                float sum = 0f;
                int qOffset = n * blocksPerRow * (blockSize / 2);
                int scaleOffset = n * blocksPerRow;
                for (int blk = 0; blk < blocksPerRow; blk++) {
                    float scale = scales[scaleOffset + blk];
                    int zpIdx = n * blocksPerRow + blk;
                    int zpByte = zeroPoints[zpIdx / 2] & 0xFF;
                    int zp = (zpIdx % 2 == 0) ? (zpByte & 0xF) : (zpByte >>> 4);

                    int kBase = blk * blockSize;
                    int qBase = qOffset + blk * (blockSize / 2);
                    for (int j = 0; j < blockSize / 2; j++) {
                        int packed = qWeight[qBase + j] & 0xFF;
                        int w0 = (packed & 0xF) - zp;
                        int w1 = (packed >>> 4) - zp;
                        sum += x[kBase + 2 * j] * (w0 * scale);
                        sum += x[kBase + 2 * j + 1] * (w1 * scale);
                    }
                }
                y[n] += sum;
            }
        }

        /** Batch matvec: Y += X @ W^T for seqLen rows. */
        public void matmul(float[] x, float[] y, int seqLen) {
            for (int s = 0; s < seqLen; s++) {
                int xOff = s * K;
                int yOff = s * N;
                float[] xRow = new float[K];
                System.arraycopy(x, xOff, xRow, 0, K);
                float[] yRow = new float[N];
                matvec(xRow, yRow);
                for (int i = 0; i < N; i++) y[yOff + i] += yRow[i];
            }
        }
    }

    /**
     * FP16 dense weight for non-quantized matmul.
     */
    public record DenseWeight(float[] data, int N, int K) {
        /** Compute y += x @ W^T where W is [N, K]. */
        public void matvec(float[] x, float[] y) {
            for (int n = 0; n < N; n++) {
                float sum = 0f;
                int offset = n * K;
                for (int k = 0; k < K; k++) {
                    sum += data[offset + k] * x[k];
                }
                y[n] += sum;
            }
        }

        /** Batch matvec. */
        public void matmul(float[] x, float[] y, int seqLen) {
            for (int s = 0; s < seqLen; s++) {
                int xOff = s * K;
                int yOff = s * N;
                for (int n = 0; n < N; n++) {
                    float sum = 0f;
                    int wOff = n * K;
                    for (int k = 0; k < K; k++) {
                        sum += data[wOff + k] * x[xOff + k];
                    }
                    y[yOff + n] += sum;
                }
            }
        }
    }

    /**
     * Abstraction over INT4 or FP16 weight matrix.
     */
    public sealed interface WeightMatrix permits QuantizedWeightMatrix, DenseWeightMatrix {
        void matvec(float[] x, float[] y);
        void matmul(float[] x, float[] y, int seqLen);
        int N();
        int K();
    }

    public record QuantizedWeightMatrix(QuantizedWeight inner) implements WeightMatrix {
        @Override public void matvec(float[] x, float[] y) { inner.matvec(x, y); }
        @Override public void matmul(float[] x, float[] y, int seqLen) { inner.matmul(x, y, seqLen); }
        @Override public int N() { return inner.N; }
        @Override public int K() { return inner.K; }
    }

    public record DenseWeightMatrix(DenseWeight inner) implements WeightMatrix {
        @Override public void matvec(float[] x, float[] y) { inner.matvec(x, y); }
        @Override public void matmul(float[] x, float[] y, int seqLen) { inner.matmul(x, y, seqLen); }
        @Override public int N() { return inner.N; }
        @Override public int K() { return inner.K; }
    }

    /**
     * Weights for a single decoder layer.
     */
    public record LayerWeights(
            float[] inputNormWeight,     // [hiddenSize]
            WeightMatrix qProj,          // [qSize, hiddenSize]
            WeightMatrix kProj,          // [kvSize, hiddenSize]
            WeightMatrix vProj,          // [kvSize, hiddenSize]
            WeightMatrix oProj,          // [hiddenSize, qSize]
            float[] postNormWeight,      // [hiddenSize]
            WeightMatrix gateProj,       // [intermediateSize, hiddenSize]
            WeightMatrix upProj,         // [intermediateSize, hiddenSize]
            WeightMatrix downProj,       // [hiddenSize, intermediateSize]
            float[] qBias,              // [qSize] or null
            float[] kBias,              // [kvSize] or null
            float[] vBias               // [kvSize] or null
    ) {
        /** Constructor without biases for backward compatibility. */
        public LayerWeights(float[] inputNormWeight, WeightMatrix qProj, WeightMatrix kProj,
                            WeightMatrix vProj, WeightMatrix oProj, float[] postNormWeight,
                            WeightMatrix gateProj, WeightMatrix upProj, WeightMatrix downProj) {
            this(inputNormWeight, qProj, kProj, vProj, oProj, postNormWeight,
                    gateProj, upProj, downProj, null, null, null);
        }
    }

    // ── Instance fields ──────────────────────────────────────────────────

    private final Qwen2Config config;
    private final RandomAccessFile raf;
    private final FileChannel channel;
    private final MappedByteBuffer externalData;

    /** Token embedding: float32 [vocabSize, hiddenSize], converted from fp16. */
    public final float[] embedTokens;

    /** Per-layer weights, indexed 0..numHiddenLayers-1. */
    public final LayerWeights[] layers;

    /** Final RMSNorm weight: float32 [hiddenSize]. */
    public final float[] finalNormWeight;

    /** LM head projection. */
    public final WeightMatrix lmHead;

    // ── External-data tensor metadata ────────────────────────────────────

    private record ExternalTensorRef(String name, int dataType, long[] dims,
                                     long offset, long length) {}

    // ── Constructor ──────────────────────────────────────────────────────

    private Qwen2Weights(Qwen2Config config, RandomAccessFile raf, FileChannel channel,
                         MappedByteBuffer externalData,
                         float[] embedTokens, LayerWeights[] layers,
                         float[] finalNormWeight, WeightMatrix lmHead) {
        this.config = config;
        this.raf = raf;
        this.channel = channel;
        this.externalData = externalData;
        this.embedTokens = embedTokens;
        this.layers = layers;
        this.finalNormWeight = finalNormWeight;
        this.lmHead = lmHead;
    }

    // ── Loading ──────────────────────────────────────────────────────────

    /**
     * Load weights from the ONNX model directory.
     *
     * @param modelDir directory containing model.onnx and model.onnx.data
     * @param config   model configuration
     * @throws IOException if files are missing or corrupt
     */
    public static Qwen2Weights load(Path modelDir, Qwen2Config config) throws IOException {
        Path onnxPath = modelDir.resolve("model.onnx");
        if (!Files.exists(onnxPath)) {
            throw new IOException("Required file missing: model.onnx (looked in " + modelDir + ")");
        }
        Path dataPath = QwenModelDirValidator.resolveExternalDataPath(modelDir);

        log.info("Loading ONNX graph from {}", onnxPath);
        OnnxGraph graph = OnnxModelReader.parse(onnxPath);

        log.info("Memory-mapping external data: {} ({} bytes)", dataPath, dataPath.toFile().length());
        RandomAccessFile raf = new RandomAccessFile(dataPath.toFile(), "r");
        FileChannel channel = null;
        try {
            channel = raf.getChannel();
            long fileSize = channel.size();
            MappedByteBuffer extData = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);
            extData.order(ByteOrder.LITTLE_ENDIAN);

            Map<String, ExternalTensorRef> externalRefs = parseExternalRefs(onnxPath);
            Map<String, OnnxTensor> inlineTensors = graph.initializers();

            // Map MatMulNBits outputs to weight tensor names
            Map<String, String[]> matmulWeightNames = new LinkedHashMap<>();
            for (OnnxNode node : graph.nodes()) {
                if ("MatMulNBits".equals(node.opType())) {
                    String output = node.outputs().get(0);
                    matmulWeightNames.put(output, new String[]{
                            node.inputs().get(1),  // Q data
                            node.inputs().get(2),  // scale
                            node.inputs().size() > 3 ? node.inputs().get(3) : null  // zero point (optional)
                    });
                }
            }

            boolean isQuantized = !matmulWeightNames.isEmpty();
            validateExternalTensorTypes(externalRefs, matmulWeightNames, isQuantized);
            log.info("Model format: {}", isQuantized ? "INT4 quantized (MatMulNBits)" : "dense FLOAT16/FLOAT");

            // ── Embedding ────────────────────────────────────────────────
            float[] embedTokens = loadEmbedding(config, externalRefs, inlineTensors, extData);
            log.info("Loaded embedding: [{}, {}]", config.vocabSize(), config.hiddenSize());

            // ── Layers ───────────────────────────────────────────────────
            LayerWeights[] layerWeights = new LayerWeights[config.numHiddenLayers()];
            for (int l = 0; l < config.numHiddenLayers(); l++) {
                layerWeights[l] = loadLayer(l, config, graph, externalRefs, inlineTensors, matmulWeightNames, extData, isQuantized);
                if ((l + 1) % 8 == 0 || l == config.numHiddenLayers() - 1) {
                    log.info("Loaded {}/{} layers", l + 1, config.numHiddenLayers());
                }
            }

            // ── Final norm ───────────────────────────────────────────────
            float[] finalNormWeight = loadFinalNormWeight(config, externalRefs, inlineTensors, extData);
            log.info("Loaded final norm weight");

            // ── LM head ─────────────────────────────────────────────────
            WeightMatrix lmHead;
            if (config.tieWordEmbeddings() && !isQuantized) {
                // Qwen2 0.5B (and most small variants) tie lm_head to embed_tokens.
                // The HuggingFace ONNX export omits lm_head.weight as a separate
                // tensor in this case — reuse the embedding matrix directly.
                log.info("Using tied lm_head (tie_word_embeddings=true): reusing embed_tokens [{}, {}]",
                        config.vocabSize(), config.hiddenSize());
                lmHead = new DenseWeightMatrix(new DenseWeight(embedTokens, config.vocabSize(), config.hiddenSize()));
            } else {
                lmHead = loadLmHead(config, graph, matmulWeightNames, externalRefs, inlineTensors, extData, isQuantized);
                log.info("Loaded lm_head: [{}, {}]", config.vocabSize(), config.hiddenSize());
            }

            log.info("All Qwen2 weights loaded successfully");
            return new Qwen2Weights(config, raf, channel, extData,
                    embedTokens, layerWeights, finalNormWeight, lmHead);
        } catch (IOException | RuntimeException e) {
            if (channel != null) {
                try {
                    channel.close();
                } catch (IOException closeError) {
                    e.addSuppressed(closeError);
                }
            }
            try {
                raf.close();
            } catch (IOException closeError) {
                e.addSuppressed(closeError);
            }
            throw e;
        }
    }

    // ── Embedding loading ────────────────────────────────────────────────

    private static float[] loadEmbedding(Qwen2Config config,
                                          Map<String, ExternalTensorRef> externalRefs,
                                          Map<String, OnnxTensor> inlineTensors,
                                          MappedByteBuffer extData) throws IOException {
        // Try common Qwen embedding weight names
        float[] data = loadOptionalTensorWithAlternatives(
                List.of("model.embed_tokens.weight", "embed_tokens.weight"),
                externalRefs, inlineTensors, extData
        );
        if (data == null) {
            throw new IOException("Embedding weight not found. Expected 'model.embed_tokens.weight' in ONNX external data.");
        }
        return data;
    }

    // ── Per-layer loading ────────────────────────────────────────────────

    private static LayerWeights loadLayer(int layerIdx, Qwen2Config config,
                                          OnnxGraph graph,
                                          Map<String, ExternalTensorRef> externalRefs,
                                          Map<String, OnnxTensor> inlineTensors,
                                          Map<String, String[]> matmulWeightNames,
                                          MappedByteBuffer extData,
                                          boolean isQuantized) throws IOException {
        String prefix = "model.layers." + layerIdx;

        // Norm weights (external fp16) — try HF-style then ONNX Community style
        float[] inputNorm = loadNormWeightWithAlternatives(prefix, "input_layernorm", externalRefs, inlineTensors, extData);
        float[] postNorm = loadNormWeightWithAlternatives(prefix, "post_attention_layernorm", externalRefs, inlineTensors, extData);

        WeightMatrix qProj, kProj, vProj, oProj, gateProj, upProj, downProj;
        float[] qBias = null, kBias = null, vBias = null;

        if (isQuantized) {
            // Load from MatMulNBits graph structure
            qProj = loadQuantizedProjection(layerIdx, "q_proj", graph, matmulWeightNames, externalRefs, extData);
            kProj = loadQuantizedProjection(layerIdx, "k_proj", graph, matmulWeightNames, externalRefs, extData);
            vProj = loadQuantizedProjection(layerIdx, "v_proj", graph, matmulWeightNames, externalRefs, extData);
            oProj = loadQuantizedProjection(layerIdx, "o_proj", graph, matmulWeightNames, externalRefs, extData);
            gateProj = loadQuantizedProjection(layerIdx, "gate_proj", graph, matmulWeightNames, externalRefs, extData);
            upProj = loadQuantizedProjection(layerIdx, "up_proj", graph, matmulWeightNames, externalRefs, extData);
            downProj = loadQuantizedProjection(layerIdx, "down_proj", graph, matmulWeightNames, externalRefs, extData);
        } else {
            // Detect naming convention: HF-style vs ONNX Community style
            boolean onnxCommunity = isOnnxCommunityFormat(prefix, externalRefs, inlineTensors);

            if (onnxCommunity) {
                // ONNX Community format: model.layers.N.attn.X_proj.MatMul.weight
                // Weights are stored as [in, out] (ONNX natural format for Y = X @ W)
                qProj = loadDenseProjectionTransposed(prefix + ".attn.q_proj.MatMul.weight", config.qSize(), config.hiddenSize(), externalRefs, inlineTensors, extData);
                kProj = loadDenseProjectionTransposed(prefix + ".attn.k_proj.MatMul.weight", config.kvSize(), config.hiddenSize(), externalRefs, inlineTensors, extData);
                vProj = loadDenseProjectionTransposed(prefix + ".attn.v_proj.MatMul.weight", config.kvSize(), config.hiddenSize(), externalRefs, inlineTensors, extData);
                oProj = loadDenseProjectionTransposed(prefix + ".attn.o_proj.MatMul.weight", config.hiddenSize(), config.qSize(), externalRefs, inlineTensors, extData);
                gateProj = loadDenseProjectionTransposed(prefix + ".mlp.gate_proj.MatMul.weight", config.intermediateSize(), config.hiddenSize(), externalRefs, inlineTensors, extData);
                upProj = loadDenseProjectionTransposed(prefix + ".mlp.up_proj.MatMul.weight", config.intermediateSize(), config.hiddenSize(), externalRefs, inlineTensors, extData);
                downProj = loadDenseProjectionTransposed(prefix + ".mlp.down_proj.MatMul.weight", config.hiddenSize(), config.intermediateSize(), externalRefs, inlineTensors, extData);

                // Load optional bias tensors (ONNX Community: model.layers.N.attn.X_proj.Add.bias)
                qBias = loadOptionalBias(prefix + ".attn.q_proj.Add.bias", externalRefs, inlineTensors, extData, config.qSize());
                kBias = loadOptionalBias(prefix + ".attn.k_proj.Add.bias", externalRefs, inlineTensors, extData, config.kvSize());
                vBias = loadOptionalBias(prefix + ".attn.v_proj.Add.bias", externalRefs, inlineTensors, extData, config.kvSize());
            } else {
                // HF-style format: model.layers.N.self_attn.X_proj.weight
                // Weights are stored as [out, in] (PyTorch convention)
                qProj = loadDenseProjection(prefix + ".self_attn.q_proj.weight", config.qSize(), config.hiddenSize(), externalRefs, inlineTensors, extData);
                kProj = loadDenseProjection(prefix + ".self_attn.k_proj.weight", config.kvSize(), config.hiddenSize(), externalRefs, inlineTensors, extData);
                vProj = loadDenseProjection(prefix + ".self_attn.v_proj.weight", config.kvSize(), config.hiddenSize(), externalRefs, inlineTensors, extData);
                oProj = loadDenseProjection(prefix + ".self_attn.o_proj.weight", config.hiddenSize(), config.qSize(), externalRefs, inlineTensors, extData);
                gateProj = loadDenseProjection(prefix + ".mlp.gate_proj.weight", config.intermediateSize(), config.hiddenSize(), externalRefs, inlineTensors, extData);
                upProj = loadDenseProjection(prefix + ".mlp.up_proj.weight", config.intermediateSize(), config.hiddenSize(), externalRefs, inlineTensors, extData);
                downProj = loadDenseProjection(prefix + ".mlp.down_proj.weight", config.hiddenSize(), config.intermediateSize(), externalRefs, inlineTensors, extData);

                // HF-style may also have biases (model.layers.N.self_attn.X_proj.bias)
                qBias = loadOptionalBias(prefix + ".self_attn.q_proj.bias", externalRefs, inlineTensors, extData, config.qSize());
                kBias = loadOptionalBias(prefix + ".self_attn.k_proj.bias", externalRefs, inlineTensors, extData, config.kvSize());
                vBias = loadOptionalBias(prefix + ".self_attn.v_proj.bias", externalRefs, inlineTensors, extData, config.kvSize());
            }
        }

        if (qBias != null && layerIdx == 0) {
            log.info("Attention biases found (q={}, k={}, v={})",
                    qBias.length, kBias != null ? kBias.length : 0, vBias != null ? vBias.length : 0);
        }

        return new LayerWeights(inputNorm, qProj, kProj, vProj, oProj, postNorm,
                gateProj, upProj, downProj, qBias, kBias, vBias);
    }

    /**
     * Detect whether the export uses ONNX Community naming (model.layers.N.attn.X_proj.MatMul.weight)
     * vs HuggingFace naming (model.layers.N.self_attn.X_proj.weight).
     */
    private static boolean isOnnxCommunityFormat(String layerPrefix,
                                                 Map<String, ExternalTensorRef> externalRefs,
                                                 Map<String, OnnxTensor> inlineTensors) {
        return tensorExists(layerPrefix + ".attn.q_proj.MatMul.weight", externalRefs, inlineTensors);
    }

    // ── Quantized weight loading ─────────────────────────────────────────

    private static WeightMatrix loadQuantizedProjection(int layerIdx, String projName,
                                                        OnnxGraph graph,
                                                        Map<String, String[]> matmulWeightNames,
                                                        Map<String, ExternalTensorRef> externalRefs,
                                                        MappedByteBuffer extData) throws IOException {
        // Find the MatMulNBits node whose output contains this projection name and layer
        String layerPrefix = "layers." + layerIdx;
        for (Map.Entry<String, String[]> entry : matmulWeightNames.entrySet()) {
            String output = entry.getKey();
            if (output.contains(layerPrefix) && output.contains(projName)) {
                QuantizedWeight qw = loadQuantizedWeight(entry.getValue(), externalRefs, extData);
                return new QuantizedWeightMatrix(qw);
            }
        }
        throw new IOException("Quantized weight not found for layer " + layerIdx + " " + projName);
    }

    private static QuantizedWeight loadQuantizedWeight(String[] tensorNames,
                                                       Map<String, ExternalTensorRef> externalRefs,
                                                       MappedByteBuffer extData) throws IOException {
        String qName = tensorNames[0];
        String sName = tensorNames[1];
        String zpName = tensorNames[2];

        ExternalTensorRef qRef = externalRefs.get(qName);
        ExternalTensorRef sRef = externalRefs.get(sName);

        if (qRef == null) throw new IOException("Quantized weight not found: " + qName);
        if (sRef == null) throw new IOException("Scale not found: " + sName);

        byte[] qData = readBytes(extData, qRef.offset, (int) qRef.length);
        float[] scales = readFloatTensorAsFloat32(extData, sRef, sName);

        byte[] zeroPoints;
        if (zpName != null) {
            ExternalTensorRef zpRef = externalRefs.get(zpName);
            if (zpRef != null) {
                zeroPoints = readBytes(extData, zpRef.offset, (int) zpRef.length);
            } else {
                int numBlocks = scales.length;
                zeroPoints = new byte[(numBlocks + 1) / 2];
                // Default zero point: 0x88 = packed (8,8) meaning zero_point=8 for both
                // nibbles in a uint4 pair — the standard symmetric quantization midpoint.
                Arrays.fill(zeroPoints, (byte) 0x88);
            }
        } else {
            int numBlocks = scales.length;
            zeroPoints = new byte[(numBlocks + 1) / 2];
            // Default zero point: 0x88 = packed (8,8), symmetric uint4 midpoint
            Arrays.fill(zeroPoints, (byte) 0x88);
        }

        int N = (int) qRef.dims[0];
        int blocksPerRow = (int) qRef.dims[1];
        int blockSize = (int) qRef.dims[2] * 2;
        int K = blocksPerRow * blockSize;

        return new QuantizedWeight(qData, scales, zeroPoints, N, K, blockSize);
    }

    // ── Dense weight loading ─────────────────────────────────────────────

    private static WeightMatrix loadDenseProjection(String tensorName, int N, int K,
                                                    Map<String, ExternalTensorRef> externalRefs,
                                                    Map<String, OnnxTensor> inlineTensors,
                                                    MappedByteBuffer extData) throws IOException {
        TensorData tensor = resolveFloatTensor(tensorName, externalRefs, inlineTensors, extData);
        if (tensor == null) {
            throw new IOException("Dense weight not found: " + tensorName);
        }
        validateMatrixShape(tensorName, tensor.dims(), N, K);
        return new DenseWeightMatrix(new DenseWeight(tensor.data(), N, K));
    }

    /**
     * Load a dense projection from ONNX Community format where weights are stored
     * in [K, N] layout (ONNX natural: Y = X @ W). Transposes to [N, K] for the
     * runtime which computes y = W @ x (equivalent to y = x @ W^T with W as [N, K]).
     */
    private static WeightMatrix loadDenseProjectionTransposed(String tensorName, int N, int K,
                                                              Map<String, ExternalTensorRef> externalRefs,
                                                              Map<String, OnnxTensor> inlineTensors,
                                                              MappedByteBuffer extData) throws IOException {
        TensorData tensor = resolveFloatTensor(tensorName, externalRefs, inlineTensors, extData);
        if (tensor == null) {
            throw new IOException("Dense weight not found: " + tensorName
                    + ". Available tensors containing 'weight': "
                    + tensorKeysContaining("weight", externalRefs, inlineTensors).stream().limit(5).toList());
        }
        float[] rawData = tensor.data();

        // Verify dimensions match expectation [K, N] from the ONNX file
        long[] dims = tensor.dims();
        if (dims.length == 2 && dims[0] == K && dims[1] == N) {
            // Expected [K, N] layout — transpose to [N, K]
            float[] transposed = new float[N * K];
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    transposed[n * K + k] = rawData[k * N + n];
                }
            }
            return new DenseWeightMatrix(new DenseWeight(transposed, N, K));
        } else if (dims.length == 2 && dims[0] == N && dims[1] == K) {
            // Already [N, K] — no transpose needed
            return new DenseWeightMatrix(new DenseWeight(rawData, N, K));
        } else {
            throw new IOException("Unexpected dimensions for " + tensorName + ": "
                    + Arrays.toString(dims) + ", expected [" + K + ", " + N + "] or [" + N + ", " + K + "]");
        }
    }

    /**
     * Load an optional bias tensor. Returns null if the tensor is not present.
     */
    private static float[] loadOptionalBias(String tensorName,
                                            Map<String, ExternalTensorRef> externalRefs,
                                            Map<String, OnnxTensor> inlineTensors,
                                            MappedByteBuffer extData,
                                            int expectedSize) throws IOException {
        TensorData tensor = resolveFloatTensor(tensorName, externalRefs, inlineTensors, extData);
        if (tensor == null) {
            return null;
        }
        validateVectorShape(tensorName, tensor.dims(), tensor.data().length, expectedSize);
        return tensor.data();
    }

    // ── Norm weight loading ──────────────────────────────────────────────

    private static float[] loadNormWeight(String tensorName,
                                          Map<String, ExternalTensorRef> externalRefs,
                                          Map<String, OnnxTensor> inlineTensors,
                                          MappedByteBuffer extData) throws IOException {
        TensorData tensor = resolveFloatTensor(tensorName, externalRefs, inlineTensors, extData);
        if (tensor == null) {
            throw new IOException("Norm weight not found: " + tensorName);
        }
        return tensor.data();
    }

    /**
     * Load per-layer norm weights, trying HF-style and ONNX Community style names.
     * HF: prefix + ".input_layernorm.weight"
     * ONNX Community: prefix + ".input_layernorm.weight" (same) or alternative patterns
     */
    private static float[] loadNormWeightWithAlternatives(String layerPrefix, String normName,
                                                          Map<String, ExternalTensorRef> externalRefs,
                                                          Map<String, OnnxTensor> inlineTensors,
                                                          MappedByteBuffer extData) throws IOException {
        // Standard HF-style name
        String primary = layerPrefix + "." + normName + ".weight";
        float[] primaryData = loadOptionalTensor(primary, externalRefs, inlineTensors, extData);
        if (primaryData != null) {
            return primaryData;
        }

        // ONNX Community alternative: LayerNormalization variant
        String altLayerNorm = layerPrefix + "." + normName + "_layernorm.weight";
        float[] altData = loadOptionalTensor(altLayerNorm, externalRefs, inlineTensors, extData);
        if (altData != null) {
            return altData;
        }

        throw new IOException("Norm weight not found: tried '" + primary + "' and '" + altLayerNorm
                + "'. Available norm-like tensors: "
                + tensorKeysMatching(k -> k.contains(layerPrefix) && (k.contains("norm") || k.contains("Norm")),
                externalRefs, inlineTensors));
    }

    /**
     * Load the final RMSNorm weight, trying multiple naming conventions.
     * HF: "model.norm.weight"
     * ONNX Community: "model.layers.{numLayers}.final_norm_layernorm.weight" or "model.norm.weight"
     */
    private static float[] loadFinalNormWeight(Qwen2Config config,
                                               Map<String, ExternalTensorRef> externalRefs,
                                               Map<String, OnnxTensor> inlineTensors,
                                               MappedByteBuffer extData) throws IOException {
        // HF-style
        float[] norm = loadOptionalTensor("model.norm.weight", externalRefs, inlineTensors, extData);
        if (norm != null) {
            return norm;
        }

        // ONNX Community: final_norm stored as a pseudo-layer after the last actual layer
        String onnxCommunityName = "model.layers." + config.numHiddenLayers() + ".final_norm_layernorm.weight";
        norm = loadOptionalTensor(onnxCommunityName, externalRefs, inlineTensors, extData);
        if (norm != null) {
            return norm;
        }

        // Try without the layer index prefix
        norm = loadOptionalTensor("model.final_norm_layernorm.weight", externalRefs, inlineTensors, extData);
        if (norm != null) {
            return norm;
        }

        throw new IOException("Final norm weight not found. Tried: 'model.norm.weight', '"
                + onnxCommunityName + "'. Available norm-like tensors: "
                + tensorKeysMatching(k -> k.contains("norm") && !k.contains("layers.0"),
                externalRefs, inlineTensors));
    }

    // ── LM Head loading ──────────────────────────────────────────────────

    private static WeightMatrix loadLmHead(Qwen2Config config, OnnxGraph graph,
                                           Map<String, String[]> matmulWeightNames,
                                           Map<String, ExternalTensorRef> externalRefs,
                                           Map<String, OnnxTensor> inlineTensors,
                                           MappedByteBuffer extData,
                                           boolean isQuantized) throws IOException {
        if (isQuantized) {
            // Find the MatMulNBits node that produces logits or lm_head output
            for (Map.Entry<String, String[]> entry : matmulWeightNames.entrySet()) {
                String output = entry.getKey();
                if (output.contains("lm_head") || output.equals("logits")) {
                    QuantizedWeight qw = loadQuantizedWeight(entry.getValue(), externalRefs, extData);
                    return new QuantizedWeightMatrix(qw);
                }
            }
            throw new IOException("LM head quantized weight not found in ONNX graph");
        } else {
            // Dense FP16 lm_head — try multiple naming patterns
            // ONNX Community: may use lm_head.MatMul.weight
            String onnxCommunityName = "lm_head.MatMul.weight";
            TensorData tensor = resolveFirstFloatTensor(
                    List.of("lm_head.weight", "model.lm_head.weight", onnxCommunityName),
                    externalRefs, inlineTensors, extData
            );
            if (tensor == null) {
                throw new IOException("LM head weight not found. Tried: 'lm_head.weight', "
                        + "'model.lm_head.weight', '" + onnxCommunityName + "' in ONNX tensors.");
            }

            float[] data = tensor.data();
            long[] dims = tensor.dims();
            int vocabSize = config.vocabSize();
            int hiddenSize = config.hiddenSize();

            // Check if transposed: ONNX Community stores as [hiddenSize, vocabSize]
            if (dims.length == 2 && dims[0] == hiddenSize && dims[1] == vocabSize) {
                // Transpose [hiddenSize, vocabSize] → [vocabSize, hiddenSize]
                float[] transposed = new float[vocabSize * hiddenSize];
                for (int v = 0; v < vocabSize; v++) {
                    for (int h = 0; h < hiddenSize; h++) {
                        transposed[v * hiddenSize + h] = data[h * vocabSize + v];
                    }
                }
                return new DenseWeightMatrix(new DenseWeight(transposed, vocabSize, hiddenSize));
            }
            if (!(dims.length == 2 && dims[0] == vocabSize && dims[1] == hiddenSize)) {
                throw new IOException("Unexpected dimensions for LM head: " + Arrays.toString(dims)
                        + ", expected [" + hiddenSize + ", " + vocabSize + "] or ["
                        + vocabSize + ", " + hiddenSize + "]");
            }
            return new DenseWeightMatrix(new DenseWeight(data, vocabSize, hiddenSize));
        }
    }

    /** Tensor payload and shape resolved from either external data refs or inline initializers. */
    private record TensorData(float[] data, long[] dims) {}

    private static TensorData resolveFirstFloatTensor(List<String> names,
                                                      Map<String, ExternalTensorRef> externalRefs,
                                                      Map<String, OnnxTensor> inlineTensors,
                                                      MappedByteBuffer extData) throws IOException {
        for (String name : names) {
            TensorData tensor = resolveFloatTensor(name, externalRefs, inlineTensors, extData);
            if (tensor != null) {
                return tensor;
            }
        }
        return null;
    }

    private static float[] loadOptionalTensor(String tensorName,
                                              Map<String, ExternalTensorRef> externalRefs,
                                              Map<String, OnnxTensor> inlineTensors,
                                              MappedByteBuffer extData) throws IOException {
        TensorData tensor = resolveFloatTensor(tensorName, externalRefs, inlineTensors, extData);
        return tensor != null ? tensor.data() : null;
    }

    private static float[] loadOptionalTensorWithAlternatives(List<String> names,
                                                              Map<String, ExternalTensorRef> externalRefs,
                                                              Map<String, OnnxTensor> inlineTensors,
                                                              MappedByteBuffer extData) throws IOException {
        TensorData tensor = resolveFirstFloatTensor(names, externalRefs, inlineTensors, extData);
        return tensor != null ? tensor.data() : null;
    }

    private static TensorData resolveFloatTensor(String tensorName,
                                                 Map<String, ExternalTensorRef> externalRefs,
                                                 Map<String, OnnxTensor> inlineTensors,
                                                 MappedByteBuffer extData) throws IOException {
        ExternalTensorRef ref = externalRefs.get(tensorName);
        if (ref != null) {
            return new TensorData(readFloatTensorAsFloat32(extData, ref, tensorName), ref.dims);
        }
        OnnxTensor inline = inlineTensors.get(tensorName);
        if (hasInlineData(inline)) {
            return new TensorData(readInlineTensorAsFloat32(inline, tensorName), inline.dims());
        }
        return null;
    }

    private static float[] readInlineTensorAsFloat32(OnnxTensor tensor, String tensorName) throws IOException {
        if (tensor.dataType() == OnnxModelReader.ONNX_FLOAT) {
            if (tensor.data().length > 0) {
                return tensor.data();
            }
            if (tensor.rawBytes().length > 0) {
                ByteBuffer bb = ByteBuffer.wrap(tensor.rawBytes()).order(ByteOrder.LITTLE_ENDIAN);
                float[] result = new float[tensor.rawBytes().length / 4];
                for (int i = 0; i < result.length; i++) {
                    result[i] = bb.getFloat();
                }
                return result;
            }
        } else if (tensor.dataType() == OnnxModelReader.ONNX_FLOAT16 && tensor.rawBytes().length > 0) {
            ByteBuffer bb = ByteBuffer.wrap(tensor.rawBytes()).order(ByteOrder.LITTLE_ENDIAN);
            float[] result = new float[tensor.rawBytes().length / 2];
            for (int i = 0; i < result.length; i++) {
                result[i] = fp16ToFp32(bb.getShort());
            }
            return result;
        }
        throw new IOException("Unsupported inline tensor data for " + tensorName + ": "
                + onnxTypeName(tensor.dataType()) + " (" + tensor.dataType() + ")");
    }

    private static void validateMatrixShape(String tensorName, long[] dims, int N, int K) throws IOException {
        if (!(dims.length == 2 && dims[0] == N && dims[1] == K)) {
            throw new IOException("Unexpected dimensions for " + tensorName + ": "
                    + Arrays.toString(dims) + ", expected [" + N + ", " + K + "]");
        }
    }

    private static void validateVectorShape(String tensorName, long[] dims, int length, int expectedSize) throws IOException {
        if (length != expectedSize) {
            throw new IOException("Unexpected size for " + tensorName + ": " + length
                    + ", expected " + expectedSize + " (dims=" + Arrays.toString(dims) + ")");
        }
    }

    private static boolean tensorExists(String tensorName,
                                        Map<String, ExternalTensorRef> externalRefs,
                                        Map<String, OnnxTensor> inlineTensors) {
        OnnxTensor inline = inlineTensors.get(tensorName);
        return externalRefs.containsKey(tensorName)
                || hasInlineData(inline);
    }

    private static Set<String> tensorKeysContaining(String fragment,
                                                    Map<String, ExternalTensorRef> externalRefs,
                                                    Map<String, OnnxTensor> inlineTensors) {
        return tensorKeysMatching(name -> name.contains(fragment), externalRefs, inlineTensors);
    }

    private static Set<String> tensorKeysMatching(Predicate<String> predicate,
                                                  Map<String, ExternalTensorRef> externalRefs,
                                                  Map<String, OnnxTensor> inlineTensors) {
        Set<String> names = new LinkedHashSet<>();
        externalRefs.keySet().stream().filter(predicate).forEach(names::add);
        inlineTensors.entrySet().stream()
                .filter(e -> hasInlineData(e.getValue()) && predicate.test(e.getKey()))
                .map(Map.Entry::getKey)
                .forEach(names::add);
        return names;
    }

    private static boolean hasInlineData(OnnxTensor tensor) {
        return tensor != null && (tensor.data().length > 0 || tensor.rawBytes().length > 0);
    }

    // ── FP16 reading ─────────────────────────────────────────────────────

    private static float[] readFloatTensorAsFloat32(MappedByteBuffer extData,
                                                    ExternalTensorRef ref,
                                                    String tensorName) throws IOException {
        if (ref == null) throw new IOException("External tensor ref is null for " + tensorName);
        if (ref.dataType == OnnxModelReader.ONNX_FLOAT16) {
            return readFp16Floats(extData, ref.offset, (int) ref.length);
        }
        if (ref.dataType == OnnxModelReader.ONNX_FLOAT) {
            return readFp32Floats(extData, ref.offset, (int) ref.length);
        }
        throw new IOException("Unsupported tensor data type for " + tensorName + ": "
                + onnxTypeName(ref.dataType) + " (" + ref.dataType + "). Supported: FLOAT16, FLOAT");
    }

    private static float[] readFp16Floats(MappedByteBuffer buf, long offset, int length) {
        int count = length / 2;
        float[] result = new float[count];
        int pos = (int) offset;
        for (int i = 0; i < count; i++) {
            short bits = buf.getShort(pos + i * 2);
            result[i] = fp16ToFp32(bits);
        }
        return result;
    }

    private static byte[] readBytes(MappedByteBuffer buf, long offset, int length) {
        byte[] result = new byte[length];
        int pos = (int) offset;
        for (int i = 0; i < length; i++) {
            result[i] = buf.get(pos + i);
        }
        return result;
    }

    private static float[] readFp32Floats(MappedByteBuffer buf, long offset, int lengthInBytes) {
        int count = lengthInBytes / 4;
        float[] result = new float[count];
        int pos = (int) offset;
        for (int i = 0; i < count; i++) {
            result[i] = buf.getFloat(pos + i * 4);
        }
        return result;
    }

    /** Convert IEEE 754 half-precision (fp16) to single-precision (fp32). */
    static float fp16ToFp32(short bits) {
        int s = (bits >>> 15) & 1;
        int e = (bits >>> 10) & 0x1F;
        int m = bits & 0x3FF;

        if (e == 0) {
            if (m == 0) return s == 0 ? 0f : -0f;
            return Float.intBitsToFloat((s << 31) | (m << 13)) * (1.0f / 16384.0f);
        } else if (e == 31) {
            return Float.intBitsToFloat((s << 31) | 0x7F800000 | (m << 13));
        } else {
            return Float.intBitsToFloat((s << 31) | ((e - 15 + 127) << 23) | (m << 13));
        }
    }

    // ── ONNX external data metadata parsing ──────────────────────────────

    private static Map<String, ExternalTensorRef> parseExternalRefs(Path onnxFile) throws IOException {
        byte[] bytes = Files.readAllBytes(onnxFile);
        ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        Map<String, ExternalTensorRef> refs = new LinkedHashMap<>();

        while (buf.hasRemaining()) {
            int loopStart = buf.position();
            int tag = readVarint32(buf);
            int fieldNum = tag >>> 3;
            int wireType = tag & 0x7;
            if (fieldNum == 7 && wireType == 2) {
                int len = readVarint32(buf);
                int end = Math.min(buf.position() + len, buf.limit());
                parseGraphForExternalRefs(buf, end, refs);
                buf.position(end);
            } else {
                skipField(buf, wireType);
            }
            if (buf.position() == loopStart) break;
        }
        log.info("Found {} external tensor references", refs.size());
        return refs;
    }

    private static void parseGraphForExternalRefs(ByteBuffer buf, int end,
                                                   Map<String, ExternalTensorRef> refs) {
        while (buf.position() < end) {
            int loopStart = buf.position();
            int tag = readVarint32(buf);
            int fieldNum = tag >>> 3;
            int wireType = tag & 0x7;
            if (fieldNum == 5 && wireType == 2) {
                int len = readVarint32(buf);
                int tEnd = Math.min(buf.position() + len, buf.limit());
                ExternalTensorRef ref = parseTensorForExternalRef(buf, tEnd);
                buf.position(tEnd);
                if (ref != null) refs.put(ref.name, ref);
            } else {
                skipField(buf, wireType);
            }
            if (buf.position() == loopStart) break;
        }
    }

    private static ExternalTensorRef parseTensorForExternalRef(ByteBuffer buf, int end) {
        List<Long> dims = new ArrayList<>();
        int dataType = 0;
        String name = "";
        int dataLocation = 0;
        long extOffset = 0;
        long extLength = 0;

        while (buf.position() < end) {
            int loopStart = buf.position();
            int tag = readVarint32(buf);
            int fieldNum = tag >>> 3;
            int wireType = tag & 0x7;
            switch (fieldNum) {
                case 1 -> {
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        int pEnd = Math.min(buf.position() + len, buf.limit());
                        while (buf.position() < pEnd) {
                            int innerStart = buf.position();
                            dims.add(readVarint64(buf));
                            if (buf.position() == innerStart) break;
                        }
                    } else if (wireType == 0) dims.add(readVarint64(buf));
                    else skipField(buf, wireType);
                }
                case 2 -> {
                    if (wireType == 0) dataType = readVarint32(buf);
                    else skipField(buf, wireType);
                }
                case 8 -> {
                    if (wireType == 2) name = readString(buf);
                    else skipField(buf, wireType);
                }
                case 13 -> {
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        int eEnd = Math.min(buf.position() + len, buf.limit());
                        String[] kv = parseStringStringEntry(buf, eEnd);
                        buf.position(eEnd);
                        if (kv != null) {
                            switch (kv[0]) {
                                case "offset" -> extOffset = Long.parseLong(kv[1]);
                                case "length" -> extLength = Long.parseLong(kv[1]);
                            }
                        }
                    } else skipField(buf, wireType);
                }
                case 14 -> {
                    if (wireType == 0) {
                        dataLocation = readVarint32(buf);
                    } else if (wireType == 2) {
                        int len = readVarint32(buf);
                        int eEnd = Math.min(buf.position() + len, buf.limit());
                        String[] kv = parseStringStringEntry(buf, eEnd);
                        buf.position(eEnd);
                        if (kv != null) {
                            switch (kv[0]) {
                                case "offset" -> extOffset = Long.parseLong(kv[1]);
                                case "length" -> extLength = Long.parseLong(kv[1]);
                            }
                        }
                    } else skipField(buf, wireType);
                }
                case 15 -> {
                    if (wireType == 0) dataLocation = readVarint32(buf);
                    else skipField(buf, wireType);
                }
                default -> skipField(buf, wireType);
            }
            if (buf.position() == loopStart) break;
        }

        if (dataLocation == 1 && !name.isEmpty()) {
            return new ExternalTensorRef(name, dataType,
                    dims.stream().mapToLong(Long::longValue).toArray(),
                    extOffset, extLength);
        }
        return null;
    }

    private static String[] parseStringStringEntry(ByteBuffer buf, int end) {
        String key = null, value = null;
        while (buf.position() < end) {
            int loopStart = buf.position();
            int tag = readVarint32(buf);
            int fieldNum = tag >>> 3;
            int wireType = tag & 0x7;
            if (fieldNum == 1 && wireType == 2) key = readString(buf);
            else if (fieldNum == 2 && wireType == 2) value = readString(buf);
            else skipField(buf, wireType);
            if (buf.position() == loopStart) break;
        }
        return (key != null && value != null) ? new String[]{key, value} : null;
    }

    // ── Protobuf primitives ──────────────────────────────────────────────

    private static int readVarint32(ByteBuffer buf) {
        int result = 0, shift = 0;
        while (buf.hasRemaining()) {
            byte b = buf.get();
            result |= (b & 0x7F) << shift;
            if ((b & 0x80) == 0) return result;
            shift += 7;
        }
        return result;
    }

    private static long readVarint64(ByteBuffer buf) {
        long result = 0;
        int shift = 0;
        while (buf.hasRemaining()) {
            byte b = buf.get();
            result |= (long) (b & 0x7F) << shift;
            if ((b & 0x80) == 0) return result;
            shift += 7;
        }
        return result;
    }

    private static String readString(ByteBuffer buf) {
        int len = readVarint32(buf);
        byte[] bytes = new byte[len];
        buf.get(bytes);
        return new String(bytes, java.nio.charset.StandardCharsets.UTF_8);
    }

    private static void skipField(ByteBuffer buf, int wireType) {
        switch (wireType) {
            case 0 -> { while (buf.hasRemaining() && (buf.get() & 0x80) != 0) {} }
            case 1 -> buf.position(Math.min(buf.position() + 8, buf.limit()));
            case 2 -> {
                int len = readVarint32(buf);
                buf.position(Math.min(buf.position() + len, buf.limit()));
            }
            case 5 -> buf.position(Math.min(buf.position() + 4, buf.limit()));
            default -> {}
        }
    }

    static String describeUnsupportedFormat(Path modelDir) {
        Path onnxPath = modelDir.resolve("model.onnx");
        if (!Files.exists(onnxPath)) {
            return "Required file missing: model.onnx (looked in " + modelDir + ")";
        }
        try {
            OnnxGraph graph = OnnxModelReader.parse(onnxPath);
            Map<String, ExternalTensorRef> refs = parseExternalRefs(onnxPath);

            Map<String, String[]> matmulWeightNames = new LinkedHashMap<>();
            for (OnnxNode node : graph.nodes()) {
                if ("MatMulNBits".equals(node.opType())) {
                    String output = node.outputs().get(0);
                    matmulWeightNames.put(output, new String[]{
                            node.inputs().get(1),
                            node.inputs().get(2),
                            node.inputs().size() > 3 ? node.inputs().get(3) : null
                    });
                }
            }
            validateExternalTensorTypes(refs, matmulWeightNames, !matmulWeightNames.isEmpty());
            return null;
        } catch (IOException | RuntimeException e) {
            String msg = e.getMessage();
            if (msg == null || msg.isBlank()) {
                // Walk to the root cause so we don't lose useful framing (e.g. when
                // the parser wraps a BufferUnderflowException with byte-offset info).
                Throwable root = e;
                while (root.getCause() != null && root.getCause() != root) root = root.getCause();
                String rootMsg = root.getMessage();
                msg = e.getClass().getSimpleName()
                        + (rootMsg != null && !rootMsg.isBlank()
                        ? " (" + rootMsg + ")"
                        : " (" + root.getClass().getSimpleName() + " — no detail; model may be truncated or use an unsupported ONNX layout)");
            }
            log.warn("Qwen model directory at {} is not a supported ONNX layout", modelDir, e);
            return "Unsupported Qwen ONNX format: " + msg;
        }
    }

    private static void validateExternalTensorTypes(Map<String, ExternalTensorRef> externalRefs,
                                                    Map<String, String[]> matmulWeightNames,
                                                    boolean isQuantized) throws IOException {
        if (externalRefs.isEmpty()) {
            throw new IOException("No external tensor references found in model.onnx");
        }
        if (isQuantized) {
            for (String[] tensors : matmulWeightNames.values()) {
                requireDataType(externalRefs, tensors[0], "quantized weight", OnnxModelReader.ONNX_UINT8, OnnxModelReader.ONNX_INT8);
                requireDataType(externalRefs, tensors[1], "quantized scale", OnnxModelReader.ONNX_FLOAT16, OnnxModelReader.ONNX_FLOAT);
                if (tensors[2] != null) {
                    requireDataType(externalRefs, tensors[2], "quantized zero-point", OnnxModelReader.ONNX_UINT8, OnnxModelReader.ONNX_INT8);
                }
            }
        } else {
            for (ExternalTensorRef ref : externalRefs.values()) {
                if (ref.name.endsWith(".weight")) {
                    if (ref.dataType != OnnxModelReader.ONNX_FLOAT16 && ref.dataType != OnnxModelReader.ONNX_FLOAT) {
                        throw new IOException("Tensor '" + ref.name + "' has unsupported data type "
                                + onnxTypeName(ref.dataType) + " (" + ref.dataType + "). Supported: FLOAT16, FLOAT");
                    }
                }
            }
        }
    }

    private static void requireDataType(Map<String, ExternalTensorRef> externalRefs,
                                        String tensorName,
                                        String tensorKind,
                                        int allowed1,
                                        int allowed2) throws IOException {
        ExternalTensorRef ref = externalRefs.get(tensorName);
        if (ref == null) {
            throw new IOException("Required " + tensorKind + " tensor not found: " + tensorName);
        }
        if (ref.dataType != allowed1 && ref.dataType != allowed2) {
            throw new IOException("Tensor '" + tensorName + "' has unsupported data type "
                    + onnxTypeName(ref.dataType) + " (" + ref.dataType + ")");
        }
    }

    private static String onnxTypeName(int dataType) {
        return switch (dataType) {
            case OnnxModelReader.ONNX_FLOAT -> "FLOAT";
            case OnnxModelReader.ONNX_FLOAT16 -> "FLOAT16";
            case OnnxModelReader.ONNX_UINT8 -> "UINT8";
            case OnnxModelReader.ONNX_INT8 -> "INT8";
            case OnnxModelReader.ONNX_INT32 -> "INT32";
            case OnnxModelReader.ONNX_INT64 -> "INT64";
            default -> "UNKNOWN";
        };
    }

    @Override
    public void close() throws IOException {
        if (channel != null && channel.isOpen()) channel.close();
        if (raf != null) raf.close();
    }
}
