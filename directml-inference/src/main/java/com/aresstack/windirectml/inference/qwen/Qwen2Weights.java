package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.windows.OnnxModelReader;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxGraph;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxNode;
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
            WeightMatrix downProj        // [hiddenSize, intermediateSize]
    ) {}

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
        Path dataPath = modelDir.resolve("model.onnx.data");

        if (!Files.exists(onnxPath)) {
            throw new IOException("Required file missing: model.onnx (looked in " + modelDir + ")");
        }
        if (!Files.exists(dataPath)) {
            throw new IOException("Required file missing: model.onnx.data (looked in " + modelDir + ")");
        }

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
            log.info("Model format: {}", isQuantized ? "INT4 quantized (MatMulNBits)" : "FP16 dense");

            // ── Embedding ────────────────────────────────────────────────
            float[] embedTokens = loadEmbedding(config, externalRefs, extData);
            log.info("Loaded embedding: [{}, {}]", config.vocabSize(), config.hiddenSize());

            // ── Layers ───────────────────────────────────────────────────
            LayerWeights[] layerWeights = new LayerWeights[config.numHiddenLayers()];
            for (int l = 0; l < config.numHiddenLayers(); l++) {
                layerWeights[l] = loadLayer(l, config, graph, externalRefs, matmulWeightNames, extData, isQuantized);
                if ((l + 1) % 8 == 0 || l == config.numHiddenLayers() - 1) {
                    log.info("Loaded {}/{} layers", l + 1, config.numHiddenLayers());
                }
            }

            // ── Final norm ───────────────────────────────────────────────
            float[] finalNormWeight = loadNormWeight("model.norm.weight", externalRefs, extData);
            log.info("Loaded final norm weight");

            // ── LM head ─────────────────────────────────────────────────
            WeightMatrix lmHead = loadLmHead(config, graph, matmulWeightNames, externalRefs, extData, isQuantized);
            log.info("Loaded lm_head: [{}, {}]", config.vocabSize(), config.hiddenSize());

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
                                          MappedByteBuffer extData) throws IOException {
        // Try common Qwen embedding weight names
        ExternalTensorRef ref = externalRefs.get("model.embed_tokens.weight");
        if (ref == null) {
            ref = externalRefs.get("embed_tokens.weight");
        }
        if (ref == null) {
            throw new IOException("Embedding weight not found. Expected 'model.embed_tokens.weight' in ONNX external data.");
        }
        return readFp16AsFloat32(extData, ref);
    }

    // ── Per-layer loading ────────────────────────────────────────────────

    private static LayerWeights loadLayer(int layerIdx, Qwen2Config config,
                                          OnnxGraph graph,
                                          Map<String, ExternalTensorRef> externalRefs,
                                          Map<String, String[]> matmulWeightNames,
                                          MappedByteBuffer extData,
                                          boolean isQuantized) throws IOException {
        String prefix = "model.layers." + layerIdx;

        // Norm weights (external fp16)
        float[] inputNorm = loadNormWeight(prefix + ".input_layernorm.weight", externalRefs, extData);
        float[] postNorm = loadNormWeight(prefix + ".post_attention_layernorm.weight", externalRefs, extData);

        WeightMatrix qProj, kProj, vProj, oProj, gateProj, upProj, downProj;

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
            // Load from external fp16 tensors
            qProj = loadDenseProjection(prefix + ".self_attn.q_proj.weight", config.qSize(), config.hiddenSize(), externalRefs, extData);
            kProj = loadDenseProjection(prefix + ".self_attn.k_proj.weight", config.kvSize(), config.hiddenSize(), externalRefs, extData);
            vProj = loadDenseProjection(prefix + ".self_attn.v_proj.weight", config.kvSize(), config.hiddenSize(), externalRefs, extData);
            oProj = loadDenseProjection(prefix + ".self_attn.o_proj.weight", config.hiddenSize(), config.qSize(), externalRefs, extData);
            gateProj = loadDenseProjection(prefix + ".mlp.gate_proj.weight", config.intermediateSize(), config.hiddenSize(), externalRefs, extData);
            upProj = loadDenseProjection(prefix + ".mlp.up_proj.weight", config.intermediateSize(), config.hiddenSize(), externalRefs, extData);
            downProj = loadDenseProjection(prefix + ".mlp.down_proj.weight", config.hiddenSize(), config.intermediateSize(), externalRefs, extData);
        }

        return new LayerWeights(inputNorm, qProj, kProj, vProj, oProj, postNorm, gateProj, upProj, downProj);
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
        float[] scales = readFp16Floats(extData, sRef.offset, (int) sRef.length);

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
                                                    MappedByteBuffer extData) throws IOException {
        ExternalTensorRef ref = externalRefs.get(tensorName);
        if (ref == null) {
            throw new IOException("Dense weight not found: " + tensorName);
        }
        float[] data = readFp16AsFloat32(extData, ref);
        return new DenseWeightMatrix(new DenseWeight(data, N, K));
    }

    // ── Norm weight loading ──────────────────────────────────────────────

    private static float[] loadNormWeight(String tensorName,
                                          Map<String, ExternalTensorRef> externalRefs,
                                          MappedByteBuffer extData) throws IOException {
        ExternalTensorRef ref = externalRefs.get(tensorName);
        if (ref == null) {
            throw new IOException("Norm weight not found: " + tensorName);
        }
        return readFp16AsFloat32(extData, ref);
    }

    // ── LM Head loading ──────────────────────────────────────────────────

    private static WeightMatrix loadLmHead(Qwen2Config config, OnnxGraph graph,
                                           Map<String, String[]> matmulWeightNames,
                                           Map<String, ExternalTensorRef> externalRefs,
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
            // Dense FP16 lm_head
            ExternalTensorRef ref = externalRefs.get("lm_head.weight");
            if (ref == null) {
                // Qwen may tie embeddings (shared weight). Check for model.lm_head.weight
                ref = externalRefs.get("model.lm_head.weight");
            }
            if (ref == null) {
                throw new IOException("LM head weight not found. Expected 'lm_head.weight' in ONNX external data.");
            }
            float[] data = readFp16AsFloat32(extData, ref);
            return new DenseWeightMatrix(new DenseWeight(data, config.vocabSize(), config.hiddenSize()));
        }
    }

    // ── FP16 reading ─────────────────────────────────────────────────────

    private static float[] readFp16AsFloat32(MappedByteBuffer extData,
                                              ExternalTensorRef ref) throws IOException {
        if (ref == null) throw new IOException("External tensor ref is null");
        return readFp16Floats(extData, ref.offset, (int) ref.length);
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
            int tag = readVarint32(buf);
            int fieldNum = tag >>> 3;
            int wireType = tag & 0x7;
            if (fieldNum == 7 && wireType == 2) {
                int len = readVarint32(buf);
                int end = buf.position() + len;
                parseGraphForExternalRefs(buf, end, refs);
                buf.position(end);
            } else {
                skipField(buf, wireType);
            }
        }
        log.info("Found {} external tensor references", refs.size());
        return refs;
    }

    private static void parseGraphForExternalRefs(ByteBuffer buf, int end,
                                                   Map<String, ExternalTensorRef> refs) {
        while (buf.position() < end) {
            int tag = readVarint32(buf);
            int fieldNum = tag >>> 3;
            int wireType = tag & 0x7;
            if (fieldNum == 5 && wireType == 2) {
                int len = readVarint32(buf);
                int tEnd = buf.position() + len;
                ExternalTensorRef ref = parseTensorForExternalRef(buf, tEnd);
                buf.position(tEnd);
                if (ref != null) refs.put(ref.name, ref);
            } else {
                skipField(buf, wireType);
            }
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
            int tag = readVarint32(buf);
            int fieldNum = tag >>> 3;
            int wireType = tag & 0x7;
            switch (fieldNum) {
                case 1 -> {
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        int pEnd = buf.position() + len;
                        while (buf.position() < pEnd) dims.add(readVarint64(buf));
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
                        int eEnd = buf.position() + len;
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
                        int eEnd = buf.position() + len;
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
            int tag = readVarint32(buf);
            int fieldNum = tag >>> 3;
            int wireType = tag & 0x7;
            if (fieldNum == 1 && wireType == 2) key = readString(buf);
            else if (fieldNum == 2 && wireType == 2) value = readString(buf);
            else skipField(buf, wireType);
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
            case 1 -> buf.position(buf.position() + 8);
            case 2 -> { int len = readVarint32(buf); buf.position(buf.position() + len); }
            case 5 -> buf.position(buf.position() + 4);
            default -> {}
        }
    }

    @Override
    public void close() throws IOException {
        if (channel != null && channel.isOpen()) channel.close();
        if (raf != null) raf.close();
    }
}
