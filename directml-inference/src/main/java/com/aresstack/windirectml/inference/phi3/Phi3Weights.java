package com.aresstack.windirectml.inference.phi3;

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
import java.nio.file.Path;
import java.util.*;
import java.util.stream.IntStream;

/**
 * Reads Phi-3-mini weights from the ONNX model graph + external data file.
 *
 * <p>The ONNX model is used only as a weight container. The graph structure
 * is parsed to discover tensor names, shapes, and data locations. Actual
 * weight bytes are read from the external {@code model.onnx.data} file
 * via memory mapping.
 *
 * <p>Weight formats:
 * <ul>
 *   <li>float16 (dt=10): norm weights, embedding, cos/sin cache, activation scales</li>
 *   <li>uint8 packed INT4 (dt=2): quantized projection weights (MatMulNBits format)</li>
 * </ul>
 */
public final class Phi3Weights implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(Phi3Weights.class);

    // ── Public types ─────────────────────────────────────────────────────

    /**
     * INT4 quantized weight block for MatMulNBits.
     * <p>
     * Format: each byte in {@code qWeight} contains 2 uint4 values (low nibble first).
     * {@code scales} are per-block float32 scales (converted from fp16 at load).
     * {@code zeroPoints} are per-block uint4 zero points packed as uint8.
     *
     * @param N         output features
     * @param K         input features
     * @param blockSize quantization block size (128)
     */
    public record QuantizedWeight(
            byte[] qWeight,    // [N, K/blockSize, blockSize/2]
            float[] scales,    // [N, K/blockSize] (fp32, converted from fp16)
            byte[] zeroPoints, // packed uint4 zero points
            int N, int K, int blockSize
    ) {
        /**
         * Dequantize and compute y = x @ W^T where W is this quantized weight.
         *
         * @param x   input vector [K]
         * @param y   output vector [N] (accumulated, not zeroed)
         */
        /**
         * Thread-local dequantization buffer (max blockSize = 256).
         */
        private static final ThreadLocal<float[]> BLOCK_BUF =
                ThreadLocal.withInitial(() -> new float[256]);

        /**
         * Dequantize and compute y += x @ W^T — parallel over output rows.
         *
         * <p><b>Prio-2 SIMD path:</b> dequantizes each quantisation block into a
         * thread-local {@code float[]} buffer, then delegates the inner dot product
         * to {@link SimdOps#dot} (Java Vector API — AVX2 = 8 lanes, AVX-512 = 16 lanes).
         * Separating the scalar nibble-extraction from the SIMD FMA step yields
         * ≈2–4× speedup over the all-scalar implementation on the hot decode path.
         */
        public void matvec(float[] x, float[] y) {
            final int blocksPerRow = K / blockSize;
            final byte[] qw = qWeight;
            final float[] sc = scales;
            final byte[] zpArr = zeroPoints;
            final int bs = blockSize;
            IntStream.range(0, N).parallel().forEach(n -> {
                float sum = 0f;
                int qOffset = n * blocksPerRow * (bs / 2);
                int scaleOffset = n * blocksPerRow;
                float[] wBuf = BLOCK_BUF.get();   // thread-local — zero GC pressure
                for (int blk = 0; blk < blocksPerRow; blk++) {
                    float scale = sc[scaleOffset + blk];
                    int zpIdx = n * blocksPerRow + blk;
                    int zpByte = zpArr[zpIdx / 2] & 0xFF;
                    float zpVal = (float) ((zpIdx % 2 == 0) ? (zpByte & 0xF) : (zpByte >>> 4));
                    int kBase = blk * bs;
                    int qBase = qOffset + blk * (bs / 2);
                    // Scalar dequant → wBuf: sequential byte reads, JIT-friendly
                    for (int j = 0; j < bs / 2; j++) {
                        int packed = qw[qBase + j] & 0xFF;
                        wBuf[2 * j] = ((packed & 0xF) - zpVal) * scale;
                        wBuf[2 * j + 1] = ((packed >>> 4) - zpVal) * scale;
                    }
                    // SIMD dot product (AVX2/AVX-512 via Java Vector API)
                    sum += SimdOps.dot(wBuf, 0, x, kBase, bs);
                }
                y[n] += sum;
            });
        }

        /**
         * Dequantize and compute Y = X @ W^T for a batch of vectors.
         *
         * @param x      input matrix [seqLen, K], row-major
         * @param y      output matrix [seqLen, N], row-major (accumulated)
         * @param seqLen number of rows
         */
        public void matmul(float[] x, float[] y, int seqLen) {
            final int nLocal = N;
            final int kLocal = K;
            final int blocksPerRow = kLocal / blockSize;
            final byte[] qw = qWeight;
            final float[] sc = scales;
            final byte[] zpArr = zeroPoints;
            final int bs = blockSize;
            final int total = seqLen * nLocal;
            IntStream.range(0, total).parallel().forEach(idx -> {
                int s = idx / nLocal;
                int n = idx - s * nLocal;
                float sum = 0f;
                int xOff = s * kLocal;
                int qOffset = n * blocksPerRow * (bs / 2);
                int scaleOffset = n * blocksPerRow;
                float[] wBuf = BLOCK_BUF.get();   // thread-local — zero GC pressure
                for (int blk = 0; blk < blocksPerRow; blk++) {
                    float scale = sc[scaleOffset + blk];
                    int zpIdx = n * blocksPerRow + blk;
                    int zpByte = zpArr[zpIdx / 2] & 0xFF;
                    float zpVal = (float) ((zpIdx % 2 == 0) ? (zpByte & 0xF) : (zpByte >>> 4));
                    int kBase = blk * bs;
                    int qBase = qOffset + blk * (bs / 2);
                    for (int j = 0; j < bs / 2; j++) {
                        int packed = qw[qBase + j] & 0xFF;
                        wBuf[2 * j] = ((packed & 0xF) - zpVal) * scale;
                        wBuf[2 * j + 1] = ((packed >>> 4) - zpVal) * scale;
                    }
                    // SIMD dot product (AVX2/AVX-512 via Java Vector API)
                    sum += SimdOps.dot(wBuf, 0, x, xOff + kBase, bs);
                }
                y[s * nLocal + n] += sum;
            });
        }
    }

    /**
     * Weights for a single decoder layer.
     */
    public record LayerWeights(
            float[] inputNormWeight,     // [hiddenSize]
            QuantizedWeight qProj,       // [hiddenSize, hiddenSize]
            QuantizedWeight kProj,       // [hiddenSize, hiddenSize]
            QuantizedWeight vProj,       // [hiddenSize, hiddenSize]
            float[] attnOutScale,        // [hiddenSize] activation scale before o_proj
            QuantizedWeight oProj,       // [hiddenSize, hiddenSize]
            float[] postNormWeight,      // [hiddenSize]
            QuantizedWeight gateUpProj,  // [2*intermediateSize, hiddenSize]
            float[] mlpOutScale,         // [intermediateSize] activation scale before down_proj
            QuantizedWeight downProj     // [hiddenSize, intermediateSize]
    ) {
    }

    // ── Instance fields ──────────────────────────────────────────────────

    private final Phi3Config config;
    private final RandomAccessFile raf;
    private final FileChannel channel;
    private final MappedByteBuffer externalData;

    /**
     * Token embedding: float32 [vocabSize, hiddenSize], converted from fp16.
     */
    public final float[] embedTokens;

    /**
     * RoPE cos cache: float32 [maxPos, headDim/2], converted from fp16.
     */
    public final float[] cosCache;

    /**
     * RoPE sin cache: float32 [maxPos, headDim/2], converted from fp16.
     */
    public final float[] sinCache;

    /**
     * Per-layer weights, indexed 0..numHiddenLayers-1.
     */
    public final LayerWeights[] layers;

    /**
     * Final RMSNorm weight: float32 [hiddenSize].
     */
    public final float[] finalNormWeight;

    /**
     * LM head (logits projection): quantized [vocabSize, hiddenSize].
     */
    public final QuantizedWeight lmHead;

    // ── External-data tensor metadata ────────────────────────────────────

    /**
     * Metadata for a tensor stored in the external data file.
     */
    private record ExternalTensorRef(String name, int dataType, long[] dims,
                                     long offset, long length) {
    }

    // ── Loading ──────────────────────────────────────────────────────────

    private Phi3Weights(Phi3Config config, RandomAccessFile raf, FileChannel channel,
                        MappedByteBuffer externalData,
                        float[] embedTokens, float[] cosCache, float[] sinCache,
                        LayerWeights[] layers, float[] finalNormWeight,
                        QuantizedWeight lmHead) {
        this.config = config;
        this.raf = raf;
        this.channel = channel;
        this.externalData = externalData;
        this.embedTokens = embedTokens;
        this.cosCache = cosCache;
        this.sinCache = sinCache;
        this.layers = layers;
        this.finalNormWeight = finalNormWeight;
        this.lmHead = lmHead;
    }

    /**
     * Package-backed factory (PHI3-WDMLPACK-COMPILER-1). Builds {@code Phi3Weights} directly from already-extracted
     * records (e.g. reconstructed from a {@code .wdmlpack}), with no ONNX file handles. This is the minimal runtime
     * seam that lets {@link Phi3Runtime} consume wdmlpack-loaded weights unchanged; {@link #close()} is a no-op for
     * this path because there is nothing to unmap.
     */
    public static Phi3Weights ofRecords(Phi3Config config,
                                        float[] embedTokens, float[] cosCache, float[] sinCache,
                                        LayerWeights[] layers, float[] finalNormWeight, QuantizedWeight lmHead) {
        return new Phi3Weights(Objects.requireNonNull(config, "config"), null, null, null,
                Objects.requireNonNull(embedTokens, "embedTokens"),
                Objects.requireNonNull(cosCache, "cosCache"),
                Objects.requireNonNull(sinCache, "sinCache"),
                Objects.requireNonNull(layers, "layers"),
                Objects.requireNonNull(finalNormWeight, "finalNormWeight"),
                Objects.requireNonNull(lmHead, "lmHead"));
    }

    /**
     * Load weights from the ONNX model directory.
     *
     * @param modelDir directory containing model.onnx and model.onnx.data
     * @param config   model configuration
     */
    public static Phi3Weights load(Path modelDir, Phi3Config config) throws IOException {
        Path onnxPath = modelDir.resolve("model.onnx");
        Path dataPath = modelDir.resolve("model.onnx.data");

        log.info("Loading ONNX graph from {}", onnxPath);
        OnnxGraph graph = OnnxModelReader.parse(onnxPath);

        log.info("Memory-mapping external data: {} ({} bytes)",
                dataPath, dataPath.toFile().length());
        RandomAccessFile raf = new RandomAccessFile(dataPath.toFile(), "r");
        FileChannel channel = raf.getChannel();
        // Map the full file. For files > 2GB we may need multiple mappings,
        // but model.onnx.data is ~2.0GB which fits in a single mapping.
        long fileSize = channel.size();
        MappedByteBuffer extData = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);
        extData.order(ByteOrder.LITTLE_ENDIAN);

        // Parse the ONNX graph to build a map of tensor references
        // (both inline tensors from graph.initializers and external-data refs)
        Map<String, OnnxTensor> inlineTensors = graph.initializers();

        // We also need external-data metadata. Our OnnxModelReader stores
        // external tensors with empty data but dims intact. For external data,
        // we parse the ONNX file a second time to extract offset/length metadata.
        Map<String, ExternalTensorRef> externalRefs = parseExternalRefs(onnxPath);

        // Map node outputs to their MatMulNBits inputs (to find weight tensor names)
        Map<String, String[]> matmulWeightNames = new LinkedHashMap<>();
        for (OnnxNode node : graph.nodes()) {
            if ("MatMulNBits".equals(node.opType())) {
                String output = node.outputs().get(0);
                // inputs: [activations, Q4G128, scale, zp]
                matmulWeightNames.put(output, new String[]{
                        node.inputs().get(1),  // Q data
                        node.inputs().get(2),  // scale
                        node.inputs().get(3)   // zero point
                });
            }
        }

        // ── Embedding ────────────────────────────────────────────────
        float[] embedTokens = readFp16AsFloat32(extData, externalRefs.get("model.embed_tokens.weight"));
        log.info("Loaded embedding: [{}, {}]", config.vocabSize(), config.hiddenSize());

        // ── Cos/Sin cache ────────────────────────────────────────────
        float[] cosCache = readFp16TensorInline(inlineTensors.get("cos_cache"));
        float[] sinCache = readFp16TensorInline(inlineTensors.get("sin_cache"));
        log.info("Loaded cos/sin cache: [{}, {}]", config.maxPositionEmbeddings(), config.headDim() / 2);

        // ── Layers ───────────────────────────────────────────────────
        LayerWeights[] layerWeights = new LayerWeights[config.numHiddenLayers()];
        for (int l = 0; l < config.numHiddenLayers(); l++) {
            layerWeights[l] = loadLayer(l, config, graph, inlineTensors, externalRefs,
                    matmulWeightNames, extData);
            if ((l + 1) % 8 == 0) {
                log.info("Loaded {}/{} layers", l + 1, config.numHiddenLayers());
            }
        }

        // ── Final norm ───────────────────────────────────────────────
        float[] finalNormWeight = readFp16AsFloat32(extData, externalRefs.get("model.norm.weight"));
        log.info("Loaded final norm weight");

        // ── LM head ─────────────────────────────────────────────────
        // The last MatMulNBits node produces "logits"
        String[] lmHeadNames = matmulWeightNames.get("logits");
        QuantizedWeight lmHead = loadQuantizedWeight(lmHeadNames, externalRefs, extData);
        log.info("Loaded lm_head: [{}, {}]", config.vocabSize(), config.hiddenSize());

        log.info("All weights loaded successfully");
        return new Phi3Weights(config, raf, channel, extData,
                embedTokens, cosCache, sinCache, layerWeights, finalNormWeight, lmHead);
    }

    // ── Per-layer loading ────────────────────────────────────────────────

    private static LayerWeights loadLayer(int layerIdx, Phi3Config config,
                                          OnnxGraph graph,
                                          Map<String, OnnxTensor> inlineTensors,
                                          Map<String, ExternalTensorRef> externalRefs,
                                          Map<String, String[]> matmulWeightNames,
                                          MappedByteBuffer extData) throws IOException {
        String prefix = "model.layers." + layerIdx;

        // Norm weights (external fp16)
        float[] inputNorm = readFp16AsFloat32(extData,
                externalRefs.get(prefix + ".input_layernorm.weight"));
        float[] postNorm = readFp16AsFloat32(extData,
                externalRefs.get(prefix + ".post_attention_layernorm.weight"));

        // Find MatMulNBits for this layer by tracing the graph
        // Layer pattern: q_proj, k_proj, v_proj all take input_layernorm output
        // o_proj takes attention output
        // gate_up_proj takes post_attention_layernorm output
        // down_proj takes MLP activation output
        String lnOut = "/model/layers." + layerIdx + "/input_layernorm/Mul_1_output_0";
        String attnOut = "/model/layers." + layerIdx + "/self_attn/Reshape_3_output_0_weight_only_out";
        String postLnOut = "/model/layers." + layerIdx + "/post_attention_layernorm/Mul_1_output_0";
        String mlpOut = "/model/layers." + layerIdx + "/mlp/act/Mul_1_output_0_weight_only_out";

        // Find the q/k/v/o projections by matching MatMulNBits input+output names
        QuantizedWeight qProj = null, kProj = null, vProj = null, oProj = null;
        QuantizedWeight gateUpProj = null, downProj = null;

        for (OnnxNode node : graph.nodes()) {
            if (!"MatMulNBits".equals(node.opType())) continue;
            String firstInput = node.inputs().get(0);
            String output = node.outputs().get(0);

            if (firstInput.equals(lnOut)) {
                String[] wNames = matmulWeightNames.get(output);
                QuantizedWeight w = loadQuantizedWeight(wNames, externalRefs, extData);
                if (output.contains("q_proj")) qProj = w;
                else if (output.contains("k_proj")) kProj = w;
                else if (output.contains("v_proj")) vProj = w;
            } else if (firstInput.equals(attnOut)) {
                String[] wNames = matmulWeightNames.get(output);
                oProj = loadQuantizedWeight(wNames, externalRefs, extData);
            } else if (firstInput.equals(postLnOut)) {
                String[] wNames = matmulWeightNames.get(output);
                gateUpProj = loadQuantizedWeight(wNames, externalRefs, extData);
            } else if (firstInput.equals(mlpOut)) {
                String[] wNames = matmulWeightNames.get(output);
                downProj = loadQuantizedWeight(wNames, externalRefs, extData);
            }
        }

        Objects.requireNonNull(qProj, "q_proj not found for layer " + layerIdx);
        Objects.requireNonNull(kProj, "k_proj not found for layer " + layerIdx);
        Objects.requireNonNull(vProj, "v_proj not found for layer " + layerIdx);
        Objects.requireNonNull(oProj, "o_proj not found for layer " + layerIdx);
        Objects.requireNonNull(gateUpProj, "gate_up_proj not found for layer " + layerIdx);
        Objects.requireNonNull(downProj, "down_proj not found for layer " + layerIdx);

        // Activation scales (inline fp16 tensors)
        String attnScaleName = "/model/layers." + layerIdx
                + "/self_attn/Reshape_3_output_0_weight_only_scale";
        String mlpScaleName = "/model/layers." + layerIdx
                + "/mlp/act/Mul_1_output_0_weight_only_scale";
        float[] attnOutScale = readFp16TensorInline(inlineTensors.get(attnScaleName));
        float[] mlpOutScale = readFp16TensorInline(inlineTensors.get(mlpScaleName));

        return new LayerWeights(inputNorm, qProj, kProj, vProj, attnOutScale,
                oProj, postNorm, gateUpProj, mlpOutScale, downProj);
    }

    // ── Quantized weight loading ─────────────────────────────────────────

    private static QuantizedWeight loadQuantizedWeight(String[] tensorNames,
                                                       Map<String, ExternalTensorRef> externalRefs,
                                                       MappedByteBuffer extData) throws IOException {
        String qName = tensorNames[0];   // *_Q4G128
        String sName = tensorNames[1];   // *_scale
        String zpName = tensorNames[2];  // *_zp

        ExternalTensorRef qRef = externalRefs.get(qName);
        ExternalTensorRef sRef = externalRefs.get(sName);
        ExternalTensorRef zpRef = externalRefs.get(zpName);

        if (qRef == null) throw new IOException("Quantized weight not found: " + qName);
        if (sRef == null) throw new IOException("Scale not found: " + sName);

        // Read Q data (uint8)
        byte[] qData = readBytes(extData, qRef.offset, (int) qRef.length);

        // Read scales (fp16 -> fp32)
        float[] scales = readFp16Floats(extData, sRef.offset, (int) sRef.length);

        // Read zero points (uint8, may be in external or inline)
        byte[] zeroPoints;
        if (zpRef != null) {
            zeroPoints = readBytes(extData, zpRef.offset, (int) zpRef.length);
        } else {
            // Default zero point = 8 for 4-bit quantization
            int numBlocks = scales.length;
            zeroPoints = new byte[(numBlocks + 1) / 2];
            Arrays.fill(zeroPoints, (byte) 0x88); // zp=8 packed as (8 | 8<<4)
        }

        // Derive N and K from Q tensor dims: [N, K/blockSize, blockSize/2]
        int N = (int) qRef.dims[0];
        int blocksPerRow = (int) qRef.dims[1];
        int blockSize = (int) qRef.dims[2] * 2; // blockSize/2 stored
        int K = blocksPerRow * blockSize;

        return new QuantizedWeight(qData, scales, zeroPoints, N, K, blockSize);
    }

    // ── FP16 reading ─────────────────────────────────────────────────────

    /**
     * Read a fp16 tensor from external data and convert to fp32.
     */
    private static float[] readFp16AsFloat32(MappedByteBuffer extData,
                                             ExternalTensorRef ref) throws IOException {
        if (ref == null) throw new IOException("External tensor ref is null");
        return readFp16Floats(extData, ref.offset, (int) ref.length);
    }

    /**
     * Read fp16 values from a mapped buffer and convert to fp32.
     */
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

    /**
     * Read an inline fp16 OnnxTensor (stored as rawBytes) and convert to fp32.
     */
    private static float[] readFp16TensorInline(OnnxTensor tensor) {
        if (tensor == null) return new float[0];
        byte[] raw = tensor.rawBytes();
        if (raw.length == 0 && tensor.data().length > 0) return tensor.data();
        int count = raw.length / 2;
        float[] result = new float[count];
        ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < count; i++) {
            result[i] = fp16ToFp32(bb.getShort());
        }
        return result;
    }

    /**
     * Read raw bytes from a mapped buffer.
     */
    private static byte[] readBytes(MappedByteBuffer buf, long offset, int length) {
        byte[] result = new byte[length];
        int pos = (int) offset;
        for (int i = 0; i < length; i++) {
            result[i] = buf.get(pos + i);
        }
        return result;
    }

    /**
     * Convert IEEE 754 half-precision (fp16) to single-precision (fp32).
     */
    static float fp16ToFp32(short bits) {
        int s = (bits >>> 15) & 1;
        int e = (bits >>> 10) & 0x1F;
        int m = bits & 0x3FF;

        if (e == 0) {
            // Subnormal or zero
            if (m == 0) return s == 0 ? 0f : -0f;
            // Subnormal: 2^(-14) * (m/1024)
            return Float.intBitsToFloat((s << 31) | (m << 13)) *
                    (1.0f / 16384.0f); // 2^-14
        } else if (e == 31) {
            // Inf or NaN
            return Float.intBitsToFloat((s << 31) | 0x7F800000 | (m << 13));
        } else {
            // Normal: bias conversion 15 -> 127
            return Float.intBitsToFloat((s << 31) | ((e - 15 + 127) << 23) | (m << 13));
        }
    }

    // ── ONNX external data metadata parsing ──────────────────────────────

    /**
     * Parse the ONNX file specifically for external data references.
     * This is a lightweight second pass that only extracts tensor metadata
     * (name, dims, data_type, external_data location/offset/length).
     */
    private static Map<String, ExternalTensorRef> parseExternalRefs(Path onnxFile) throws IOException {
        byte[] bytes = java.nio.file.Files.readAllBytes(onnxFile);
        ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        Map<String, ExternalTensorRef> refs = new LinkedHashMap<>();

        // ModelProto: field 7 = graph
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
                // TensorProto initializer
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
        int dataLocation = 0;  // 0=DEFAULT, 1=EXTERNAL
        long extOffset = 0;
        long extLength = 0;

        while (buf.position() < end) {
            int tag = readVarint32(buf);
            int fieldNum = tag >>> 3;
            int wireType = tag & 0x7;
            switch (fieldNum) {
                case 1 -> { // dims
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        int pEnd = buf.position() + len;
                        while (buf.position() < pEnd) dims.add(readVarint64(buf));
                    } else if (wireType == 0) dims.add(readVarint64(buf));
                    else skipField(buf, wireType);
                }
                case 2 -> { // data_type
                    if (wireType == 0) dataType = readVarint32(buf);
                    else skipField(buf, wireType);
                }
                case 8 -> { // name
                    if (wireType == 2) name = readString(buf);
                    else skipField(buf, wireType);
                }
                case 13 -> { // external_data (repeated StringStringEntryProto) — field 13 in this ONNX variant
                    // NOTE: In this ONNX model variant, external_data is encoded as field 13
                    // (sharing the field number with raw_data). We distinguish by checking wire type:
                    // raw_data is bytes (wireType=2), StringStringEntryProto is also wireType=2 but
                    // we parse the content to detect key-value pairs.
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
                case 14 -> { // data_location enum OR external_data (depending on ONNX proto version)
                    if (wireType == 0) {
                        // data_location as varint (this ONNX variant)
                        dataLocation = readVarint32(buf);
                    } else if (wireType == 2) {
                        // external_data as StringStringEntryProto (standard ONNX spec)
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
                case 15 -> { // data_location enum (standard ONNX spec)
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
        return null; // Not external
    }

    /**
     * Parse a StringStringEntryProto: field 1 = key, field 2 = value.
     */
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

    // ── Protobuf primitives (duplicated from OnnxModelReader to avoid coupling) ──

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
            case 0 -> readVarint64(buf);
            case 1 -> buf.position(buf.position() + 8);
            case 2 -> {
                int len = readVarint32(buf);
                buf.position(buf.position() + len);
            }
            case 5 -> buf.position(buf.position() + 4);
            default -> throw new RuntimeException("Unknown wire type: " + wireType);
        }
    }

    // ── AutoCloseable ────────────────────────────────────────────────────

    @Override
    public void close() throws IOException {
        // Package-backed instances (ofRecords) hold no file handles -> nothing to unmap.
        if (channel != null) {
            try {
                channel.close();
            } catch (Exception ignored) {
            }
        }
        if (raf != null) {
            try {
                raf.close();
            } catch (Exception ignored) {
            }
        }
    }
}
