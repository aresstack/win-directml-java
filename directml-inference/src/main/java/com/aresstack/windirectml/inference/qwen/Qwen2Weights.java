package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.model.ModelSource;
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
import java.util.stream.IntStream;

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
        /**
         * Thread-local buffer for block dequantization (max blockSize = 256).
         * Avoids per-call allocation on the hot decode path.
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
            final byte[] zp = zeroPoints;
            final int bs = blockSize;
            IntStream.range(0, N).parallel().forEach(n -> {
                float sum = 0f;
                int qOffset = n * blocksPerRow * (bs / 2);
                int scaleOffset = n * blocksPerRow;
                float[] wBuf = BLOCK_BUF.get();   // thread-local — zero GC pressure
                for (int blk = 0; blk < blocksPerRow; blk++) {
                    float scale = sc[scaleOffset + blk];
                    int zpIdx = n * blocksPerRow + blk;
                    int zpByte = zp[zpIdx / 2] & 0xFF;
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
         * Batch matvec: Y += X @ W^T for seqLen rows — parallel over (s, n).
         */
        public void matmul(float[] x, float[] y, int seqLen) {
            final int nLocal = N;
            final int kLocal = K;
            final int blocksPerRow = kLocal / blockSize;
            final byte[] qw = qWeight;
            final float[] sc = scales;
            final byte[] zp = zeroPoints;
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
                    int zpByte = zp[zpIdx / 2] & 0xFF;
                    float zpVal = (float) ((zpIdx % 2 == 0) ? (zpByte & 0xF) : (zpByte >>> 4));
                    int kBase = blk * bs;
                    int qBase = qOffset + blk * (bs / 2);
                    for (int j = 0; j < bs / 2; j++) {
                        int packed = qw[qBase + j] & 0xFF;
                        wBuf[2 * j] = ((packed & 0xF) - zpVal) * scale;
                        wBuf[2 * j + 1] = ((packed >>> 4) - zpVal) * scale;
                    }
                    sum += SimdOps.dot(wBuf, 0, x, xOff + kBase, bs);
                }
                y[s * nLocal + n] += sum;
            });
        }
    }

    /**
     * FP16 dense weight for non-quantized matmul.
     */
    public record DenseWeight(float[] data, int N, int K) {
        /**
         * Compute y += x @ W^T where W is [N, K] — row-parallel over output rows.
         */
        public void matvec(float[] x, float[] y) {
            final int nLocal = N;
            final int kLocal = K;
            final float[] dLocal = data;
            IntStream.range(0, nLocal).parallel().forEach(n -> {
                y[n] += SimdOps.dot(dLocal, n * kLocal, x, 0, kLocal);
            });
        }

        /**
         * Batch matvec — parallelized over (seq, outRow) flattened index.
         */
        public void matmul(float[] x, float[] y, int seqLen) {
            final int nLocal = N;
            final int kLocal = K;
            final float[] dLocal = data;
            final int total = seqLen * nLocal;
            IntStream.range(0, total).parallel().forEach(idx -> {
                int s = idx / nLocal;
                int n = idx - s * nLocal;
                y[s * nLocal + n] += SimdOps.dot(dLocal, n * kLocal, x, s * kLocal, kLocal);
            });
        }
    }

    /**
     * Compact embedding table abstraction. The common Qwen ONNX export stores
     * embeddings as FLOAT16; keeping them as FP16 and expanding only the
     * requested rows cuts Java-heap residency for embeddings roughly in half
     * compared with the old eager float[] conversion.
     */
    public interface EmbeddingTable {
        int rows();

        int cols();

        void copyRow(int row, float[] dest, int destOffset);

        long estimatedHostBytes();

        float[] materializeFloatArray();
    }

    public record FloatEmbeddingTable(float[] data, int rows, int cols) implements EmbeddingTable {
        @Override
        public void copyRow(int row, float[] dest, int destOffset) {
            System.arraycopy(data, row * cols, dest, destOffset, cols);
        }

        @Override
        public long estimatedHostBytes() {
            return (long) data.length * Float.BYTES;
        }

        @Override
        public float[] materializeFloatArray() {
            return data;
        }
    }

    public record RawFp16EmbeddingTable(byte[] data, int rows, int cols) implements EmbeddingTable {
        @Override
        public void copyRow(int row, float[] dest, int destOffset) {
            int src = row * cols * Short.BYTES;
            for (int i = 0; i < cols; i++) {
                int p = src + i * Short.BYTES;
                short bits = (short) ((data[p] & 0xFF) | ((data[p + 1] & 0xFF) << 8));
                dest[destOffset + i] = fp16ToFp32(bits);
            }
        }

        @Override
        public long estimatedHostBytes() {
            return data.length;
        }

        @Override
        public float[] materializeFloatArray() {
            float[] out = new float[rows * cols];
            for (int r = 0; r < rows; r++) {
                copyRow(r, out, r * cols);
            }
            return out;
        }
    }

    public record MappedFp16EmbeddingTable(ByteBuffer data, long offset, int rows, int cols) implements EmbeddingTable {
        @Override
        public void copyRow(int row, float[] dest, int destOffset) {
            ByteBuffer srcBuf = data.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
            int src = Math.toIntExact(offset + (long) row * cols * Short.BYTES);
            for (int i = 0; i < cols; i++) {
                dest[destOffset + i] = fp16ToFp32(srcBuf.getShort(src + i * Short.BYTES));
            }
        }

        @Override
        public long estimatedHostBytes() {
            return 0L;
        }

        @Override
        public float[] materializeFloatArray() {
            float[] out = new float[rows * cols];
            for (int r = 0; r < rows; r++) {
                copyRow(r, out, r * cols);
            }
            return out;
        }
    }

    /**
     * Placeholder installed after GPU upload to drop heavyweight host-side
     * projection arrays. Any later CPU fallback that reaches this object is a
     * programming/configuration error and should fail loudly instead of silently
     * producing nonsense.
     */
    public record ReleasedWeightMatrix(String name, int N, int K) implements WeightMatrix {
        @Override
        public void matvec(float[] x, float[] y) {
            throw new IllegalStateException("Host weight storage has been released for " + name);
        }

        @Override
        public void matmul(float[] x, float[] y, int seqLen) {
            throw new IllegalStateException("Host weight storage has been released for " + name);
        }
    }

    /**
     * Abstraction over INT4 or FP16 weight matrix.
     */
    public sealed interface WeightMatrix permits QuantizedWeightMatrix, DenseWeightMatrix, ReleasedWeightMatrix {
        void matvec(float[] x, float[] y);

        void matmul(float[] x, float[] y, int seqLen);

        int N();

        int K();
    }

    public record QuantizedWeightMatrix(QuantizedWeight inner) implements WeightMatrix {
        @Override
        public void matvec(float[] x, float[] y) {
            inner.matvec(x, y);
        }

        @Override
        public void matmul(float[] x, float[] y, int seqLen) {
            inner.matmul(x, y, seqLen);
        }

        @Override
        public int N() {
            return inner.N;
        }

        @Override
        public int K() {
            return inner.K;
        }
    }

    public record DenseWeightMatrix(DenseWeight inner) implements WeightMatrix {
        @Override
        public void matvec(float[] x, float[] y) {
            inner.matvec(x, y);
        }

        @Override
        public void matmul(float[] x, float[] y, int seqLen) {
            inner.matmul(x, y, seqLen);
        }

        @Override
        public int N() {
            return inner.N;
        }

        @Override
        public int K() {
            return inner.K;
        }
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
        /**
         * Constructor without biases for backward compatibility.
         */
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

    /**
     * Token embedding table [vocabSize, hiddenSize]. Kept compact when the
     * source tensor is FP16 so Java does not need a permanent 500+ MB float[].
     */
    public final EmbeddingTable embedTokens;

    /**
     * Per-layer weights, indexed 0..numHiddenLayers-1.
     */
    public final LayerWeights[] layers;

    /**
     * Final RMSNorm weight: float32 [hiddenSize].
     */
    public final float[] finalNormWeight;

    /**
     * LM head projection.
     */
    public WeightMatrix lmHead;

    // ── External-data tensor metadata ────────────────────────────────────

    record ExternalTensorRef(String name, int dataType, long[] dims,
                             long offset, long length) {
    }

    // ── Constructor ──────────────────────────────────────────────────────

    private Qwen2Weights(Qwen2Config config, RandomAccessFile raf, FileChannel channel,
                         MappedByteBuffer externalData,
                         EmbeddingTable embedTokens, LayerWeights[] layers,
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
     * Load weights from the default ONNX model file in the model directory.
     *
     * @param modelDir directory containing model.onnx and, for the dense export, model.onnx_data
     * @param config   model configuration
     * @throws IOException if files are missing or corrupt
     */
    public static Qwen2Weights load(Path modelDir, Qwen2Config config) throws IOException {
        return load(modelDir, config, QwenModelDirValidator.DEFAULT_MODEL_FILE);
    }

    /**
     * Load weights from a selected ONNX model file in the model directory.
     *
     * @param modelDir      directory containing model metadata and ONNX files
     * @param config        model configuration
     * @param modelFileName ONNX filename inside modelDir
     * @throws IOException if files are missing or corrupt
     */
    public static Qwen2Weights load(Path modelDir, Qwen2Config config, String modelFileName) throws IOException {
        String safeModelFileName = QwenModelDirValidator.normalizeModelFileName(modelFileName);
        Path packagePath = QwenWdmlPackCompiler.resolveOutputPath(modelDir, safeModelFileName);
        boolean packageLoaded = false;
        ModelSource<QwenModelImport> modelSource;
        if (QwenWdmlPackCompiler.shouldLoadPackage() && Files.isRegularFile(packagePath)) {
            modelSource = new QwenWdmlPackModelSource(modelDir, packagePath, safeModelFileName, config);
            packageLoaded = true;
        } else {
            modelSource = new QwenOnnxModelSource(modelDir, safeModelFileName);
        }

        log.info("Loading Qwen model through import layer: format={}, source={}",
                modelSource.format(), modelSource.location());
        QwenModelImport imported = modelSource.load();
        OnnxGraph graph = imported.graph();
        Map<String, ExternalTensorRef> externalRefs = imported.externalRefs();
        Map<String, OnnxTensor> inlineTensors = imported.inlineTensors();
        log.info("Qwen tensor catalog: {}", imported.tensorCatalog().summary());
        if (!packageLoaded) {
            QwenWdmlPackCompiler.writeManifestIfAutoCreateEnabled(imported, config, modelDir, safeModelFileName);
            QwenWdmlPackCompiler.writeManifestIfRequested(imported, config, modelDir, safeModelFileName);
        } else {
            log.info("Qwen wdmlpack package accepted: {}", packagePath);
        }

        RandomAccessFile raf = null;
        FileChannel channel = null;
        MappedByteBuffer extData = null;
        try {
            if (!externalRefs.isEmpty()) {
                Path dataPath = QwenModelDirValidator.resolveExternalDataPath(modelDir);
                log.info("Memory-mapping external data: {} ({} bytes)", dataPath, dataPath.toFile().length());
                raf = new RandomAccessFile(dataPath.toFile(), "r");
                channel = raf.getChannel();
                long fileSize = channel.size();
                extData = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);
                extData.order(ByteOrder.LITTLE_ENDIAN);
            } else {
                log.info("No external ONNX tensor data references found; using inline initializers from {}", safeModelFileName);
            }

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
            validateTensorTypes(externalRefs, inlineTensors, matmulWeightNames, isQuantized);
            log.info("Model format: {}", isQuantized ? "INT4 quantized (MatMulNBits)" : "dense FLOAT16/FLOAT");

            // ── Embedding ────────────────────────────────────────────────
            EmbeddingTable embedTokens = loadEmbedding(config, externalRefs, inlineTensors, extData);
            log.info("Loaded embedding: [{}, {}] (hostStorage={})",
                    config.vocabSize(), config.hiddenSize(), formatBytes(embedTokens.estimatedHostBytes()));

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
            WeightMatrix lmHead = loadLmHeadOrTiedEmbedding(
                    config, graph, matmulWeightNames, externalRefs, inlineTensors, extData,
                    isQuantized, embedTokens);
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
            if (raf != null) {
                try {
                    raf.close();
                } catch (IOException closeError) {
                    e.addSuppressed(closeError);
                }
            }
            throw e;
        }
    }

    // ── Embedding loading ────────────────────────────────────────────────

    private static EmbeddingTable loadEmbedding(Qwen2Config config,
                                                Map<String, ExternalTensorRef> externalRefs,
                                                Map<String, OnnxTensor> inlineTensors,
                                                MappedByteBuffer extData) throws IOException {
        for (String name : List.of("model.embed_tokens.weight", "embed_tokens.weight")) {
            EmbeddingTable table = resolveEmbeddingTable(name, config, externalRefs, inlineTensors, extData);
            if (table != null) {
                return table;
            }
        }

        // v20 heap-light: large inline FP16 tensors may be backed by an mmap slice.
        // Some ONNX exports place huge raw_data fields before the TensorProto name,
        // and the lightweight parser can then lose the initializer key while still
        // retaining the tensor payload. Recover by matching the only FP16 matrix
        // with the exact embedding shape. This keeps the mmap-backed path without
        // falling back to a 260 MiB byte[] copy.
        EmbeddingTable byShape = findInlineEmbeddingByShape(config, inlineTensors);
        if (byShape != null) {
            return byShape;
        }

        throw new IOException("Embedding weight not found. Expected 'model.embed_tokens.weight' in ONNX data.");
    }

    private static EmbeddingTable findInlineEmbeddingByShape(Qwen2Config config,
                                                             Map<String, OnnxTensor> inlineTensors) throws IOException {
        int rows = config.vocabSize();
        int cols = config.hiddenSize();
        List<OnnxTensor> candidates = inlineTensors.entrySet().stream()
                .map(Map.Entry::getValue)
                .filter(t -> t != null
                        && t.dataType() == OnnxModelReader.ONNX_FLOAT16
                        && t.rawByteLength() > 0
                        && t.dims().length == 2
                        && t.dims()[0] == rows
                        && t.dims()[1] == cols)
                .toList();

        if (candidates.isEmpty()) {
            return null;
        }
        if (candidates.size() > 1) {
            String names = inlineTensors.entrySet().stream()
                    .filter(e -> candidates.contains(e.getValue()))
                    .map(Map.Entry::getKey)
                    .toList()
                    .toString();
            throw new IOException("Ambiguous inline FP16 embedding candidates with shape ["
                    + rows + ", " + cols + "]: " + names);
        }

        OnnxTensor tensor = candidates.get(0);
        String recoveredName = inlineTensors.entrySet().stream()
                .filter(e -> e.getValue() == tensor)
                .map(Map.Entry::getKey)
                .findFirst()
                .orElse("<unknown>");
        if (tensor.rawBytes().length > 0) {
            log.warn("Embedding tensor name lookup failed; recovered inline FP16 embedding by shape "
                    + "[{}, {}] from '{}' using compact host storage", rows, cols, recoveredName);
            return new RawFp16EmbeddingTable(tensor.rawBytes(), rows, cols);
        }
        log.warn("Embedding tensor name lookup failed; recovered mmap-backed inline FP16 embedding "
                + "by shape [{}, {}] from '{}'", rows, cols, recoveredName);
        return new MappedFp16EmbeddingTable(tensor.rawDataBuffer(), 0L, rows, cols);
    }

    private static EmbeddingTable resolveEmbeddingTable(String tensorName, Qwen2Config config,
                                                        Map<String, ExternalTensorRef> externalRefs,
                                                        Map<String, OnnxTensor> inlineTensors,
                                                        MappedByteBuffer extData) throws IOException {
        int rows = config.vocabSize();
        int cols = config.hiddenSize();
        ExternalTensorRef ref = externalRefs.get(tensorName);
        if (ref != null) {
            validateMatrixShape(tensorName, ref.dims, rows, cols);
            if (ref.dataType == OnnxModelReader.ONNX_FLOAT16) {
                if (extData == null) {
                    throw new IOException("External tensor data is unavailable for " + tensorName);
                }
                log.info("Using mmap-backed FP16 embedding table for {}", tensorName);
                return new MappedFp16EmbeddingTable(extData, ref.offset, rows, cols);
            }
            if (ref.dataType == OnnxModelReader.ONNX_FLOAT) {
                return new FloatEmbeddingTable(readFloatTensorAsFloat32(extData, ref, tensorName), rows, cols);
            }
            throw new IOException("Unsupported embedding tensor type for " + tensorName + ": "
                    + onnxTypeName(ref.dataType) + " (" + ref.dataType + ")");
        }

        OnnxTensor inline = inlineTensors.get(tensorName);
        if (!hasInlineData(inline)) {
            return null;
        }
        validateMatrixShape(tensorName, inline.dims(), rows, cols);
        if (inline.dataType() == OnnxModelReader.ONNX_FLOAT16 && inline.rawByteLength() > 0) {
            if (inline.rawBytes().length > 0) {
                log.info("Using compact FP16 embedding table for {}", tensorName);
                return new RawFp16EmbeddingTable(inline.rawBytes(), rows, cols);
            }
            log.info("Using mmap-backed inline FP16 embedding table for {}", tensorName);
            return new MappedFp16EmbeddingTable(inline.rawDataBuffer(), 0L, rows, cols);
        }
        return new FloatEmbeddingTable(readInlineTensorAsFloat32(inline, tensorName), rows, cols);
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
            // Load from MatMulNBits graph structure. Preserve Q/K/V biases because Qwen2.5
            // uses attention projection bias; avoid drifting q4f16 decode into random tokens.
            qProj = loadQuantizedProjection(layerIdx, "q_proj", graph, matmulWeightNames, externalRefs, inlineTensors, extData);
            kProj = loadQuantizedProjection(layerIdx, "k_proj", graph, matmulWeightNames, externalRefs, inlineTensors, extData);
            vProj = loadQuantizedProjection(layerIdx, "v_proj", graph, matmulWeightNames, externalRefs, inlineTensors, extData);
            oProj = loadQuantizedProjection(layerIdx, "o_proj", graph, matmulWeightNames, externalRefs, inlineTensors, extData);
            gateProj = loadQuantizedProjection(layerIdx, "gate_proj", graph, matmulWeightNames, externalRefs, inlineTensors, extData);
            upProj = loadQuantizedProjection(layerIdx, "up_proj", graph, matmulWeightNames, externalRefs, inlineTensors, extData);
            downProj = loadQuantizedProjection(layerIdx, "down_proj", graph, matmulWeightNames, externalRefs, inlineTensors, extData);

            qBias = loadOptionalQuantizedProjectionBias(layerIdx, "q_proj", graph, matmulWeightNames,
                    externalRefs, inlineTensors, extData, config.qSize());
            kBias = loadOptionalQuantizedProjectionBias(layerIdx, "k_proj", graph, matmulWeightNames,
                    externalRefs, inlineTensors, extData, config.kvSize());
            vBias = loadOptionalQuantizedProjectionBias(layerIdx, "v_proj", graph, matmulWeightNames,
                    externalRefs, inlineTensors, extData, config.kvSize());
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
                                                        Map<String, OnnxTensor> inlineTensors,
                                                        MappedByteBuffer extData) throws IOException {
        // Find the MatMulNBits node whose output contains this projection name and layer
        String layerPrefix = "layers." + layerIdx;
        for (Map.Entry<String, String[]> entry : matmulWeightNames.entrySet()) {
            String output = entry.getKey();
            if (output.contains(layerPrefix) && output.contains(projName)) {
                QuantizedWeight qw = loadQuantizedWeight(entry.getValue(), externalRefs, inlineTensors, extData);
                return new QuantizedWeightMatrix(qw);
            }
        }
        throw new IOException("Quantized weight not found for layer " + layerIdx + " " + projName);
    }

    /**
     * Load an optional Q/K/V projection bias for quantized ONNX Community graphs.
     *
     * <p>Preserve Qwen2.5 attention QKV bias. Resolve the bias through the Add node
     * after MatMulNBits instead of relying on one hard-coded tensor name.</p>
     */
    private static float[] loadOptionalQuantizedProjectionBias(int layerIdx,
                                                               String projName,
                                                               OnnxGraph graph,
                                                               Map<String, String[]> matmulWeightNames,
                                                               Map<String, ExternalTensorRef> externalRefs,
                                                               Map<String, OnnxTensor> inlineTensors,
                                                               MappedByteBuffer extData,
                                                               int expectedSize) throws IOException {
        String layerPrefix = "model.layers." + layerIdx;

        List<String> candidateNames = List.of(
                layerPrefix + ".self_attn." + projName + ".bias",
                layerPrefix + ".attn." + projName + ".bias",
                layerPrefix + ".attn." + projName + ".Add.bias",
                layerPrefix + ".self_attn." + projName + ".Add.bias"
        );
        for (String candidateName : candidateNames) {
            float[] bias = loadOptionalBiasIfShapeMatches(candidateName,
                    externalRefs, inlineTensors, extData, expectedSize);
            if (bias != null) {
                return bias;
            }
        }

        String matMulOutput = findQuantizedProjectionOutput(layerIdx, projName, matmulWeightNames);
        if (matMulOutput != null) {
            float[] connectedBias = findBiasConnectedToMatMulOutput(matMulOutput, graph,
                    externalRefs, inlineTensors, extData, expectedSize);
            if (connectedBias != null) {
                return connectedBias;
            }
        }

        return findBiasByLayerAndProjectionName(layerIdx, projName,
                externalRefs, inlineTensors, extData, expectedSize);
    }

    private static String findQuantizedProjectionOutput(int layerIdx,
                                                        String projName,
                                                        Map<String, String[]> matmulWeightNames) {
        String layerPrefix = "layers." + layerIdx;
        for (String output : matmulWeightNames.keySet()) {
            if (output.contains(layerPrefix) && output.contains(projName)) {
                return output;
            }
        }
        return null;
    }

    private static float[] findBiasConnectedToMatMulOutput(String matMulOutput,
                                                           OnnxGraph graph,
                                                           Map<String, ExternalTensorRef> externalRefs,
                                                           Map<String, OnnxTensor> inlineTensors,
                                                           MappedByteBuffer extData,
                                                           int expectedSize) throws IOException {
        for (OnnxNode node : graph.nodes()) {
            if (!"Add".equals(node.opType()) || node.inputs().size() < 2) {
                continue;
            }
            if (!node.inputs().contains(matMulOutput)) {
                continue;
            }
            for (String inputName : node.inputs()) {
                if (inputName.equals(matMulOutput)) {
                    continue;
                }
                float[] bias = loadOptionalBiasIfShapeMatches(inputName,
                        externalRefs, inlineTensors, extData, expectedSize);
                if (bias != null) {
                    return bias;
                }
            }
        }
        return null;
    }

    private static float[] findBiasByLayerAndProjectionName(int layerIdx,
                                                            String projName,
                                                            Map<String, ExternalTensorRef> externalRefs,
                                                            Map<String, OnnxTensor> inlineTensors,
                                                            MappedByteBuffer extData,
                                                            int expectedSize) throws IOException {
        String layerFragment = "layers." + layerIdx;
        String projectionFragment = projName;
        List<String> biasNames = tensorKeysMatching(name -> {
            String lower = name.toLowerCase(Locale.ROOT);
            return name.contains(layerFragment)
                    && name.contains(projectionFragment)
                    && lower.contains("bias");
        }, externalRefs, inlineTensors).stream().sorted().toList();

        for (String biasName : biasNames) {
            float[] bias = loadOptionalBiasIfShapeMatches(biasName,
                    externalRefs, inlineTensors, extData, expectedSize);
            if (bias != null) {
                return bias;
            }
        }
        return null;
    }

    private static float[] loadOptionalBiasIfShapeMatches(String tensorName,
                                                          Map<String, ExternalTensorRef> externalRefs,
                                                          Map<String, OnnxTensor> inlineTensors,
                                                          MappedByteBuffer extData,
                                                          int expectedSize) throws IOException {
        TensorData tensor = resolveFloatTensor(tensorName, externalRefs, inlineTensors, extData);
        if (tensor == null || tensor.data().length != expectedSize) {
            return null;
        }
        return tensor.data();
    }

    private static QuantizedWeight loadQuantizedWeight(String[] tensorNames,
                                                       Map<String, ExternalTensorRef> externalRefs,
                                                       Map<String, OnnxTensor> inlineTensors,
                                                       MappedByteBuffer extData) throws IOException {
        String qName = tensorNames[0];
        String sName = tensorNames[1];
        String zpName = tensorNames[2];

        RawTensorData qTensor = resolveRawTensor(qName, externalRefs, inlineTensors, extData);
        TensorData scaleTensor = resolveFloatTensor(sName, externalRefs, inlineTensors, extData);

        if (qTensor == null) {
            throw new IOException("Quantized weight not found: " + qName);
        }
        if (scaleTensor == null) {
            throw new IOException("Scale not found: " + sName);
        }

        byte[] qData = qTensor.data();
        float[] scales = scaleTensor.data();

        long[] dims = qTensor.dims();
        if (dims.length < 3) {
            throw new IOException("Unexpected dimensions for quantized weight " + qName + ": "
                    + Arrays.toString(dims) + ", expected [N, blocks, packedBlock]");
        }
        int N = (int) dims[0];
        int blocksPerRow = (int) dims[1];
        int blockSize = (int) dims[2] * 2;
        int K = blocksPerRow * blockSize;

        if (scales.length != N * blocksPerRow) {
            throw new IOException("Unexpected scale tensor size for " + sName + ": " + scales.length
                    + ", expected " + (N * blocksPerRow) + " for quantized weight " + qName);
        }

        byte[] zeroPoints = resolvePackedUInt4ZeroPoints(
                zpName, externalRefs, inlineTensors, extData, N, blocksPerRow);

        return new QuantizedWeight(qData, scales, zeroPoints, N, K, blockSize);
    }

    /**
     * Resolve MatMulNBits zero-points and normalize them to the runtime layout.
     *
     * <p>Normalize ONNX row-wise packed zero-points from
     * {@code [N, ceil(k_blocks * 4 / 8)]} into the flat continuous nibble stream
     * expected by the runtime kernel. Preserve correct row alignment when
     * {@code k_blocks} is odd.</p>
     */
    private static byte[] resolvePackedUInt4ZeroPoints(String tensorName,
                                                       Map<String, ExternalTensorRef> externalRefs,
                                                       Map<String, OnnxTensor> inlineTensors,
                                                       MappedByteBuffer extData,
                                                       int rowCount,
                                                       int blocksPerRow) throws IOException {
        int blockCount = rowCount * blocksPerRow;
        if (tensorName == null) {
            return createDefaultZeroPoints(blockCount);
        }

        Integer dataType = findTensorDataType(externalRefs, inlineTensors, tensorName);
        if (dataType == null) {
            return createDefaultZeroPoints(blockCount);
        }

        if (dataType == OnnxModelReader.ONNX_FLOAT || dataType == OnnxModelReader.ONNX_FLOAT16) {
            TensorData zeroPointTensor = resolveFloatTensor(tensorName, externalRefs, inlineTensors, extData);
            if (zeroPointTensor == null) {
                return createDefaultZeroPoints(blockCount);
            }
            return packUnpackedUInt4ZeroPoints(zeroPointTensor.data(), tensorName, rowCount, blocksPerRow);
        }

        RawTensorData rawZeroPointTensor = resolveRawTensor(tensorName, externalRefs, inlineTensors, extData);
        if (rawZeroPointTensor == null) {
            return createDefaultZeroPoints(blockCount);
        }
        return normalizeRawUInt4ZeroPoints(rawZeroPointTensor, tensorName, rowCount, blocksPerRow);
    }

    private static byte[] normalizeRawUInt4ZeroPoints(RawTensorData tensor,
                                                      String tensorName,
                                                      int rowCount,
                                                      int blocksPerRow) throws IOException {
        byte[] data = tensor.data();
        long[] dims = tensor.dims();
        int blockCount = rowCount * blocksPerRow;
        int runtimePackedBytes = (blockCount + 1) / 2;
        int rowPackedBytes = (blocksPerRow + 1) / 2;
        int rowPackedTotalBytes = rowCount * rowPackedBytes;

        if (dims.length == 2 && dims[0] == rowCount && dims[1] == rowPackedBytes) {
            return repackRowPaddedUInt4ZeroPoints(data, tensorName, rowCount, blocksPerRow, rowPackedBytes);
        }
        if (dims.length == 2 && dims[0] == rowCount && dims[1] == blocksPerRow) {
            return packUnpackedUInt4ZeroPoints(data, tensorName, rowCount, blocksPerRow);
        }
        if (data.length == rowPackedTotalBytes && rowPackedTotalBytes != runtimePackedBytes) {
            return repackRowPaddedUInt4ZeroPoints(data, tensorName, rowCount, blocksPerRow, rowPackedBytes);
        }
        if (data.length == runtimePackedBytes) {
            return data;
        }
        if (data.length == blockCount) {
            return packUnpackedUInt4ZeroPoints(data, tensorName, rowCount, blocksPerRow);
        }

        throw new IOException("Unexpected zero-point layout for " + tensorName + ": dims="
                + Arrays.toString(dims) + ", bytes=" + data.length + ", expected packed row layout ["
                + rowCount + ", " + rowPackedBytes + "], flat packed bytes " + runtimePackedBytes
                + ", or unpacked [" + rowCount + ", " + blocksPerRow + "]");
    }

    private static byte[] repackRowPaddedUInt4ZeroPoints(byte[] rowPackedZeroPoints,
                                                         String tensorName,
                                                         int rowCount,
                                                         int blocksPerRow,
                                                         int rowPackedBytes) throws IOException {
        int expectedBytes = rowCount * rowPackedBytes;
        if (rowPackedZeroPoints.length < expectedBytes) {
            throw new IOException("Zero-point tensor " + tensorName + " is truncated: "
                    + rowPackedZeroPoints.length + " bytes, expected at least " + expectedBytes);
        }

        byte[] packed = createEmptyPackedUInt4(rowCount * blocksPerRow);
        for (int row = 0; row < rowCount; row++) {
            int rowOffset = row * rowPackedBytes;
            int blockBase = row * blocksPerRow;
            for (int block = 0; block < blocksPerRow; block++) {
                int value = unpackUInt4(rowPackedZeroPoints, rowOffset * 2 + block);
                packUInt4(packed, blockBase + block, value);
            }
        }
        log.debug("Repacked row-padded zero-points for {}: rows={}, blocksPerRow={}, rowBytes={} -> {} bytes",
                tensorName, rowCount, blocksPerRow, rowPackedBytes, packed.length);
        return packed;
    }

    private static byte[] packUnpackedUInt4ZeroPoints(byte[] unpackedZeroPoints,
                                                      String tensorName,
                                                      int rowCount,
                                                      int blocksPerRow) throws IOException {
        int blockCount = rowCount * blocksPerRow;
        if (unpackedZeroPoints.length < blockCount) {
            throw new IOException("Zero-point tensor " + tensorName + " is truncated: "
                    + unpackedZeroPoints.length + " values, expected " + blockCount);
        }
        byte[] packed = createEmptyPackedUInt4(blockCount);
        for (int i = 0; i < blockCount; i++) {
            packUInt4(packed, i, unpackedZeroPoints[i] & 0x0F);
        }
        return packed;
    }

    private static byte[] packUnpackedUInt4ZeroPoints(float[] unpackedZeroPoints,
                                                      String tensorName,
                                                      int rowCount,
                                                      int blocksPerRow) throws IOException {
        int blockCount = rowCount * blocksPerRow;
        if (unpackedZeroPoints.length < blockCount) {
            throw new IOException("Zero-point tensor " + tensorName + " is truncated: "
                    + unpackedZeroPoints.length + " values, expected " + blockCount);
        }
        byte[] packed = createEmptyPackedUInt4(blockCount);
        for (int i = 0; i < blockCount; i++) {
            packUInt4(packed, i, clampUInt4(Math.round(unpackedZeroPoints[i])));
        }
        return packed;
    }

    private static byte[] createDefaultZeroPoints(int numBlocks) {
        byte[] zeroPoints = createEmptyPackedUInt4(numBlocks);
        // Default zero point: 0x88 = packed (8,8), symmetric uint4 midpoint.
        Arrays.fill(zeroPoints, (byte) 0x88);
        return zeroPoints;
    }

    private static byte[] createEmptyPackedUInt4(int valueCount) {
        return new byte[(valueCount + 1) / 2];
    }

    private static int unpackUInt4(byte[] data, int nibbleIndex) {
        int packed = data[nibbleIndex / 2] & 0xFF;
        return (nibbleIndex & 1) == 0 ? (packed & 0x0F) : (packed >>> 4);
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

    @SuppressWarnings("unused") // kept for tests and older reflective callers
    private static WeightMatrix loadLmHeadOrTiedEmbedding(Qwen2Config config, OnnxGraph graph,
                                                          Map<String, String[]> matmulWeightNames,
                                                          Map<String, ExternalTensorRef> externalRefs,
                                                          Map<String, OnnxTensor> inlineTensors,
                                                          MappedByteBuffer extData,
                                                          boolean isQuantized,
                                                          float[] embedTokens) throws IOException {
        return loadLmHeadOrTiedEmbedding(config, graph, matmulWeightNames,
                externalRefs, inlineTensors, extData, isQuantized,
                new FloatEmbeddingTable(embedTokens, config.vocabSize(), config.hiddenSize()));
    }

    private static WeightMatrix loadLmHeadOrTiedEmbedding(Qwen2Config config, OnnxGraph graph,
                                                          Map<String, String[]> matmulWeightNames,
                                                          Map<String, ExternalTensorRef> externalRefs,
                                                          Map<String, OnnxTensor> inlineTensors,
                                                          MappedByteBuffer extData,
                                                          boolean isQuantized,
                                                          EmbeddingTable embedTokens) throws IOException {
        boolean preferExplicitQuantizedLmHead = isQuantized && Boolean.parseBoolean(
                System.getProperty("qwen.tiedLmHead.preferExplicitQuantized", "true"));
        if (preferExplicitQuantizedLmHead) {
            WeightMatrix quantizedLmHead = findQuantizedLmHead(graph, matmulWeightNames,
                    externalRefs, inlineTensors, extData);
            if (quantizedLmHead != null) {
                log.info("Using explicit quantized lm_head from ONNX instead of tied dense embedding "
                                + "(tie_word_embeddings={})",
                        config.tieWordEmbeddings());
                return quantizedLmHead;
            }
        }

        if (config.tieWordEmbeddings()) {
            return createTiedLmHead(config, isQuantized, embedTokens);
        }

        WeightMatrix explicitLmHead = loadOptionalLmHead(config, graph, matmulWeightNames,
                externalRefs, inlineTensors, extData, isQuantized);
        if (explicitLmHead != null) {
            return explicitLmHead;
        }
        throw new IOException("LM head weight not found in ONNX graph and tie_word_embeddings=false");
    }

    private static WeightMatrix loadOptionalLmHead(Qwen2Config config, OnnxGraph graph,
                                                   Map<String, String[]> matmulWeightNames,
                                                   Map<String, ExternalTensorRef> externalRefs,
                                                   Map<String, OnnxTensor> inlineTensors,
                                                   MappedByteBuffer extData,
                                                   boolean isQuantized) throws IOException {
        if (isQuantized) {
            WeightMatrix quantizedLmHead = findQuantizedLmHead(graph, matmulWeightNames,
                    externalRefs, inlineTensors, extData);
            if (quantizedLmHead != null) {
                return quantizedLmHead;
            }
        }
        return findDenseLmHead(config, externalRefs, inlineTensors, extData);
    }

    private static WeightMatrix createTiedLmHead(Qwen2Config config,
                                                 boolean isQuantized,
                                                 EmbeddingTable embedTokens) {
        boolean quantizeTiedLmHead = isQuantized && Boolean.parseBoolean(
                System.getProperty("qwen.tiedLmHead.quantize", "false"));
        if (quantizeTiedLmHead) {
            int blockSize = chooseTiedLmHeadBlockSize(config.hiddenSize());
            if (blockSize > 0) {
                log.info("Using tied lm_head as runtime INT4: quantizing embed_tokens [{}, {}] with blockSize={}",
                        config.vocabSize(), config.hiddenSize(), blockSize);
                QuantizedWeight quantized = quantizeDenseRowsToUInt4(
                        embedTokens.materializeFloatArray(), config.vocabSize(), config.hiddenSize(), blockSize);
                return new QuantizedWeightMatrix(quantized);
            }
            log.warn("Cannot runtime-quantize tied lm_head because hidden size {} is not block-aligned; using dense embedding",
                    config.hiddenSize());
        }

        log.info("Using tied lm_head (tie_word_embeddings=true): reusing embed_tokens [{}, {}]",
                config.vocabSize(), config.hiddenSize());
        return new DenseWeightMatrix(new DenseWeight(embedTokens.materializeFloatArray(), config.vocabSize(), config.hiddenSize()));
    }

    private static int chooseTiedLmHeadBlockSize(int hiddenSize) {
        int[] candidates = {128, 64, 32, 16};
        for (int candidate : candidates) {
            if (hiddenSize % candidate == 0) {
                return candidate;
            }
        }
        return 0;
    }

    private static QuantizedWeight quantizeDenseRowsToUInt4(float[] data, int rowCount, int columnCount, int blockSize) {
        int blocksPerRow = columnCount / blockSize;
        int rowBytes = columnCount / 2;
        byte[] qWeight = new byte[rowCount * rowBytes];
        float[] scales = new float[rowCount * blocksPerRow];
        byte[] zeroPoints = new byte[(rowCount * blocksPerRow + 1) / 2];

        for (int row = 0; row < rowCount; row++) {
            int rowOffset = row * columnCount;
            int qRowOffset = row * rowBytes;
            int blockBase = row * blocksPerRow;
            for (int block = 0; block < blocksPerRow; block++) {
                int valueOffset = rowOffset + block * blockSize;
                QuantizationBlock blockQuantization = calculateUInt4QuantizationBlock(data, valueOffset, blockSize);
                int blockIndex = blockBase + block;
                scales[blockIndex] = blockQuantization.scale();
                packUInt4(zeroPoints, blockIndex, blockQuantization.zeroPoint());
                quantizeDenseBlockToUInt4(data, valueOffset, blockSize, qWeight,
                        qRowOffset + block * (blockSize / 2), blockQuantization);
            }
        }
        return new QuantizedWeight(qWeight, scales, zeroPoints, rowCount, columnCount, blockSize);
    }

    private static QuantizationBlock calculateUInt4QuantizationBlock(float[] data, int offset, int length) {
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < length; i++) {
            float value = data[offset + i];
            if (value < min) {
                min = value;
            }
            if (value > max) {
                max = value;
            }
        }

        if (max == min) {
            if (Math.abs(max) < 1.0e-8f) {
                return new QuantizationBlock(1.0f, 0);
            }
            if (max > 0.0f) {
                return new QuantizationBlock(max / 15.0f, 0);
            }
            return new QuantizationBlock(-min / 15.0f, 15);
        }

        float scale = (max - min) / 15.0f;
        int zeroPoint = clampUInt4(Math.round(-min / scale));
        return new QuantizationBlock(scale, zeroPoint);
    }

    private static void quantizeDenseBlockToUInt4(float[] data, int inputOffset, int length,
                                                  byte[] output, int outputOffset,
                                                  QuantizationBlock quantization) {
        for (int i = 0; i < length; i++) {
            int quantized = clampUInt4(Math.round(data[inputOffset + i] / quantization.scale()
                    + quantization.zeroPoint()));
            packUInt4(output, outputOffset * 2 + i, quantized);
        }
    }

    private static void packUInt4(byte[] output, int nibbleIndex, int value) {
        int byteIndex = nibbleIndex / 2;
        int packedValue = value & 0x0F;
        if ((nibbleIndex & 1) == 0) {
            output[byteIndex] = (byte) ((output[byteIndex] & 0xF0) | packedValue);
        } else {
            output[byteIndex] = (byte) ((output[byteIndex] & 0x0F) | (packedValue << 4));
        }
    }

    private static int clampUInt4(int value) {
        if (value < 0) {
            return 0;
        }
        if (value > 15) {
            return 15;
        }
        return value;
    }

    private record QuantizationBlock(float scale, int zeroPoint) {
    }

    private static WeightMatrix findQuantizedLmHead(OnnxGraph graph,
                                                    Map<String, String[]> matmulWeightNames,
                                                    Map<String, ExternalTensorRef> externalRefs,
                                                    Map<String, OnnxTensor> inlineTensors,
                                                    MappedByteBuffer extData) throws IOException {
        for (Map.Entry<String, String[]> entry : matmulWeightNames.entrySet()) {
            String output = entry.getKey();
            String[] tensorNames = entry.getValue();
            if (looksLikeLmHeadOutput(output) || looksLikeLmHeadTensorNames(tensorNames)) {
                log.info("Using quantized lm_head from MatMulNBits output '{}' (weight={})",
                        output, tensorNames.length > 0 ? tensorNames[0] : "<missing>");
                QuantizedWeight quantizedWeight = loadQuantizedWeight(tensorNames, externalRefs, inlineTensors, extData);
                return new QuantizedWeightMatrix(quantizedWeight);
            }
        }

        return null;
    }

    private static boolean looksLikeLmHeadOutput(String outputName) {
        String lower = outputName.toLowerCase(Locale.ROOT);
        return lower.contains("lm_head") || lower.equals("logits") || lower.endsWith("/logits");
    }

    private static boolean looksLikeLmHeadTensorNames(String[] tensorNames) {
        if (tensorNames == null) {
            return false;
        }
        for (String tensorName : tensorNames) {
            if (tensorName != null && tensorName.toLowerCase(Locale.ROOT).contains("lm_head")) {
                return true;
            }
        }
        return false;
    }

    private static WeightMatrix findDenseLmHead(Qwen2Config config,
                                                Map<String, ExternalTensorRef> externalRefs,
                                                Map<String, OnnxTensor> inlineTensors,
                                                MappedByteBuffer extData) throws IOException {
        String onnxCommunityName = "lm_head.MatMul.weight";
        TensorData tensor = resolveFirstFloatTensor(
                List.of("lm_head.weight", "model.lm_head.weight", onnxCommunityName),
                externalRefs, inlineTensors, extData
        );
        if (tensor == null) {
            return null;
        }
        return createLmHeadDenseMatrix(config, tensor);
    }

    private static WeightMatrix createLmHeadDenseMatrix(Qwen2Config config, TensorData tensor) throws IOException {
        float[] data = tensor.data();
        long[] dims = tensor.dims();
        int vocabSize = config.vocabSize();
        int hiddenSize = config.hiddenSize();

        if (dims.length == 2 && dims[0] == hiddenSize && dims[1] == vocabSize) {
            float[] transposed = new float[vocabSize * hiddenSize];
            for (int vocabIndex = 0; vocabIndex < vocabSize; vocabIndex++) {
                for (int hiddenIndex = 0; hiddenIndex < hiddenSize; hiddenIndex++) {
                    transposed[vocabIndex * hiddenSize + hiddenIndex] = data[hiddenIndex * vocabSize + vocabIndex];
                }
            }
            return new DenseWeightMatrix(new DenseWeight(transposed, vocabSize, hiddenSize));
        }
        if (dims.length == 2 && dims[0] == vocabSize && dims[1] == hiddenSize) {
            return new DenseWeightMatrix(new DenseWeight(data, vocabSize, hiddenSize));
        }
        throw new IOException("Unexpected dimensions for LM head: " + Arrays.toString(dims)
                + ", expected [" + hiddenSize + ", " + vocabSize + "] or ["
                + vocabSize + ", " + hiddenSize + "]");
    }

    private static WeightMatrix loadLmHead(Qwen2Config config, OnnxGraph graph,
                                           Map<String, String[]> matmulWeightNames,
                                           Map<String, ExternalTensorRef> externalRefs,
                                           Map<String, OnnxTensor> inlineTensors,
                                           MappedByteBuffer extData,
                                           boolean isQuantized) throws IOException {
        WeightMatrix lmHead = loadOptionalLmHead(config, graph, matmulWeightNames,
                externalRefs, inlineTensors, extData, isQuantized);
        if (lmHead != null) {
            return lmHead;
        }
        throw new IOException("LM head weight not found in ONNX graph");
    }

    /**
     * Tensor float payload and shape resolved from either external data refs or inline initializers.
     */
    private record TensorData(float[] data, long[] dims) {
    }

    /**
     * Tensor raw byte payload and shape resolved from either external data refs or inline initializers.
     */
    private record RawTensorData(byte[] data, long[] dims, int dataType) {
    }

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

    private static RawTensorData resolveRawTensor(String tensorName,
                                                  Map<String, ExternalTensorRef> externalRefs,
                                                  Map<String, OnnxTensor> inlineTensors,
                                                  MappedByteBuffer extData) throws IOException {
        ExternalTensorRef ref = externalRefs.get(tensorName);
        if (ref != null) {
            if (extData == null) {
                throw new IOException("External tensor data is unavailable for " + tensorName);
            }
            return new RawTensorData(readBytes(extData, ref.offset, (int) ref.length), ref.dims, ref.dataType);
        }
        OnnxTensor inline = inlineTensors.get(tensorName);
        if (inline != null && inline.rawByteLength() > 0) {
            return new RawTensorData(inline.rawBytesOrCopy(), inline.dims(), inline.dataType());
        }
        return null;
    }

    private static TensorData resolveFloatTensor(String tensorName,
                                                 Map<String, ExternalTensorRef> externalRefs,
                                                 Map<String, OnnxTensor> inlineTensors,
                                                 MappedByteBuffer extData) throws IOException {
        ExternalTensorRef ref = externalRefs.get(tensorName);
        if (ref != null) {
            if (extData == null) {
                throw new IOException("External tensor data is unavailable for " + tensorName);
            }
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
            if (tensor.rawByteLength() > 0) {
                ByteBuffer bb = tensor.rawDataBuffer().order(ByteOrder.LITTLE_ENDIAN);
                float[] result = new float[tensor.rawByteLength() / 4];
                for (int i = 0; i < result.length; i++) {
                    result[i] = bb.getFloat();
                }
                return result;
            }
        } else if (tensor.dataType() == OnnxModelReader.ONNX_FLOAT16 && tensor.rawByteLength() > 0) {
            ByteBuffer bb = tensor.rawDataBuffer().order(ByteOrder.LITTLE_ENDIAN);
            float[] result = new float[tensor.rawByteLength() / 2];
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
        return tensor != null && (tensor.data().length > 0 || tensor.rawByteLength() > 0);
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

    /**
     * Convert IEEE 754 half-precision (fp16) to single-precision (fp32).
     */
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

    public void copyEmbedding(int tokenId, float[] dest, int destOffset) {
        embedTokens.copyRow(tokenId, dest, destOffset);
    }

    /**
     * Drop large host-side projection matrices after they have been uploaded to
     * D3D12 resources. Norms/biases/embeddings stay resident because prefill and
     * token lookup still need them on the Java side.
     */
    public long releaseGpuUploadedProjectionStorage(boolean includeLmHead) {
        long released = 0L;
        for (int i = 0; i < layers.length; i++) {
            LayerWeights lw = layers[i];
            released += estimatedBytes(lw.qProj());
            released += estimatedBytes(lw.kProj());
            released += estimatedBytes(lw.vProj());
            released += estimatedBytes(lw.oProj());
            released += estimatedBytes(lw.gateProj());
            released += estimatedBytes(lw.upProj());
            released += estimatedBytes(lw.downProj());
            layers[i] = new LayerWeights(
                    lw.inputNormWeight(),
                    new ReleasedWeightMatrix("layer." + i + ".q_proj", lw.qProj().N(), lw.qProj().K()),
                    new ReleasedWeightMatrix("layer." + i + ".k_proj", lw.kProj().N(), lw.kProj().K()),
                    new ReleasedWeightMatrix("layer." + i + ".v_proj", lw.vProj().N(), lw.vProj().K()),
                    new ReleasedWeightMatrix("layer." + i + ".o_proj", lw.oProj().N(), lw.oProj().K()),
                    lw.postNormWeight(),
                    new ReleasedWeightMatrix("layer." + i + ".gate_proj", lw.gateProj().N(), lw.gateProj().K()),
                    new ReleasedWeightMatrix("layer." + i + ".up_proj", lw.upProj().N(), lw.upProj().K()),
                    new ReleasedWeightMatrix("layer." + i + ".down_proj", lw.downProj().N(), lw.downProj().K()),
                    lw.qBias(), lw.kBias(), lw.vBias());
        }
        if (includeLmHead && !(lmHead instanceof ReleasedWeightMatrix)) {
            released += estimatedBytes(lmHead);
            lmHead = new ReleasedWeightMatrix("lm_head", lmHead.N(), lmHead.K());
        }
        if (released > 0) {
            log.info("Released host-side uploaded Qwen projection weights: {}", formatBytes(released));
        }
        return released;
    }

    private static long estimatedBytes(WeightMatrix matrix) {
        if (matrix instanceof QuantizedWeightMatrix q) {
            QuantizedWeight inner = q.inner();
            return (long) inner.qWeight().length
                    + (long) inner.scales().length * Float.BYTES
                    + (long) inner.zeroPoints().length;
        }
        if (matrix instanceof DenseWeightMatrix d) {
            return (long) d.inner().data().length * Float.BYTES;
        }
        return 0L;
    }

    static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        double kib = bytes / 1024.0;
        if (kib < 1024) return String.format(Locale.ROOT, "%.1f KiB", kib);
        double mib = kib / 1024.0;
        if (mib < 1024) return String.format(Locale.ROOT, "%.1f MiB", mib);
        return String.format(Locale.ROOT, "%.2f GiB", mib / 1024.0);
    }

    // ── ONNX external data metadata parsing ──────────────────────────────

    static Map<String, ExternalTensorRef> parseExternalRefs(Path onnxFile) throws IOException {
        try (FileChannel channel = FileChannel.open(onnxFile, java.nio.file.StandardOpenOption.READ)) {
            long size = channel.size();
            if (size > Integer.MAX_VALUE) {
                throw new IOException("ONNX model is too large for external-ref scan: "
                        + onnxFile.getFileName() + " (" + size + " bytes)");
            }
            MappedByteBuffer mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, size);
            mapped.order(ByteOrder.LITTLE_ENDIAN);
            return parseExternalRefs(mapped);
        }
    }

    private static Map<String, ExternalTensorRef> parseExternalRefs(ByteBuffer buf) {
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
            case 0 -> {
                while (buf.hasRemaining() && (buf.get() & 0x80) != 0) {
                }
            }
            case 1 -> buf.position(Math.min(buf.position() + 8, buf.limit()));
            case 2 -> {
                int len = readVarint32(buf);
                buf.position(Math.min(buf.position() + len, buf.limit()));
            }
            case 5 -> buf.position(Math.min(buf.position() + 4, buf.limit()));
            default -> {
            }
        }
    }

    static String describeUnsupportedFormat(Path modelDir) {
        return describeUnsupportedFormat(modelDir, QwenModelDirValidator.DEFAULT_MODEL_FILE);
    }

    static String describeUnsupportedFormat(Path modelDir, String modelFileName) {
        String safeModelFileName;
        try {
            safeModelFileName = QwenModelDirValidator.normalizeModelFileName(modelFileName);
        } catch (IllegalArgumentException ex) {
            return "Invalid Qwen ONNX filename: " + ex.getMessage();
        }

        Path onnxPath = modelDir.resolve(safeModelFileName);
        if (!Files.exists(onnxPath)) {
            return "Required file missing: " + safeModelFileName + " (looked in " + modelDir + ")";
        }
        try {
            OnnxGraph graph = OnnxModelReader.parse(onnxPath);
            Map<String, ExternalTensorRef> refs = parseExternalRefs(onnxPath);
            Map<String, OnnxTensor> inlineTensors = graph.initializers();

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
            validateTensorTypes(refs, inlineTensors, matmulWeightNames, !matmulWeightNames.isEmpty());
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

    private static void validateTensorTypes(Map<String, ExternalTensorRef> externalRefs,
                                            Map<String, OnnxTensor> inlineTensors,
                                            Map<String, String[]> matmulWeightNames,
                                            boolean isQuantized) throws IOException {
        if (externalRefs.isEmpty() && inlineTensors.isEmpty()) {
            throw new IOException("No ONNX tensor initializers found in selected model file");
        }
        if (isQuantized) {
            for (String[] tensors : matmulWeightNames.values()) {
                requireDataType(externalRefs, inlineTensors, tensors[0], "quantized weight",
                        OnnxModelReader.ONNX_UINT8, OnnxModelReader.ONNX_INT8);
                requireDataType(externalRefs, inlineTensors, tensors[1], "quantized scale",
                        OnnxModelReader.ONNX_FLOAT16, OnnxModelReader.ONNX_FLOAT);
                if (tensors[2] != null) {
                    requireDataType(externalRefs, inlineTensors, tensors[2], "quantized zero-point",
                            OnnxModelReader.ONNX_UINT8, OnnxModelReader.ONNX_INT8);
                }
            }
        } else {
            for (ExternalTensorRef ref : externalRefs.values()) {
                validateDenseWeightDataType(ref.name, ref.dataType);
            }
            for (Map.Entry<String, OnnxTensor> entry : inlineTensors.entrySet()) {
                OnnxTensor tensor = entry.getValue();
                if (hasInlineData(tensor)) {
                    validateDenseWeightDataType(entry.getKey(), tensor.dataType());
                }
            }
        }
    }

    private static void validateDenseWeightDataType(String tensorName, int dataType) throws IOException {
        if (tensorName.endsWith(".weight")
                && dataType != OnnxModelReader.ONNX_FLOAT16
                && dataType != OnnxModelReader.ONNX_FLOAT) {
            throw new IOException("Tensor '" + tensorName + "' has unsupported data type "
                    + onnxTypeName(dataType) + " (" + dataType + "). Supported: FLOAT16, FLOAT");
        }
    }

    private static void requireDataType(Map<String, ExternalTensorRef> externalRefs,
                                        Map<String, OnnxTensor> inlineTensors,
                                        String tensorName,
                                        String tensorKind,
                                        int allowed1,
                                        int allowed2) throws IOException {
        Integer dataType = findTensorDataType(externalRefs, inlineTensors, tensorName);
        if (dataType == null) {
            throw new IOException("Required " + tensorKind + " tensor not found: " + tensorName);
        }
        if (dataType.intValue() != allowed1 && dataType.intValue() != allowed2) {
            throw new IOException("Tensor '" + tensorName + "' has unsupported data type "
                    + onnxTypeName(dataType.intValue()) + " (" + dataType + ")");
        }
    }

    private static Integer findTensorDataType(Map<String, ExternalTensorRef> externalRefs,
                                              Map<String, OnnxTensor> inlineTensors,
                                              String tensorName) {
        ExternalTensorRef ref = externalRefs.get(tensorName);
        if (ref != null) {
            return ref.dataType;
        }
        OnnxTensor inline = inlineTensors.get(tensorName);
        if (hasInlineData(inline)) {
            return inline.dataType();
        }
        return null;
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
