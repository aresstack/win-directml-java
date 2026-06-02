package com.aresstack.windirectml.windows;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.foreign.*;
import java.nio.file.Path;
import java.util.*;

/**
 * Small 28×28 grayscale CNN DirectML inference pipeline.
 * <p>
 * V1 scope: small 28×28 grayscale CNN vertical slice, validated with
 * {@code mnist-12.onnx} (float32), {@code mnist-12-int8.onnx} (int8 quantized),
 * and {@code mnist_emnist_blank_cnn_v1.onnx} (EMNIST+blank, 11 classes).
 * Also compatible with {@code mnist-8.onnx}.
 * <p>
 * Supported architectures:
 * <ul>
 *   <li><b>MNIST</b>: Input(1,1,28,28) → Conv+Relu → MaxPool → Conv+Relu → MaxPool → Gemm → Output(1,10)</li>
 *   <li><b>EMNIST_BLANK</b>: Input(1,1,28,28) → Conv+Relu → Conv+Relu → MaxPool → Conv+Relu → MaxPool
 *       → Flatten → Gemm+Relu(BN folded) → Gemm → Output(1,11)</li>
 * </ul>
 * <p>
 * BatchNorm is supported via <b>inference-mode fusion</b>: BN parameters (scale, bias, mean, variance)
 * are folded into the preceding FC layer's weights and bias at load time. This is the correct V1
 * solution – no separate general-purpose BatchNorm runtime operator is needed.
 * <p>
 * No ONNX Runtime, no third-party libs. Pure FFM → Windows DLLs.
 */
public final class MnistPipeline implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(MnistPipeline.class);

    /**
     * Detected model architecture.
     */
    enum ModelArch {MNIST, EMNIST_BLANK}

    private final WindowsBindings wb;
    private Arena arena;
    private ModelArch arch = ModelArch.MNIST;
    private int outputSize = 10;

    // ── Parsed weights (shared) ──────────────────────────────────────────
    private float[] conv1Filter, conv1Bias;
    private float[] conv2Filter, conv2Bias;
    private float[] fcWeight, fcBias;  // final FC layer (MNIST: 256→10, EMNIST: 128→11)

    // ── EMNIST-specific parsed weights ───────────────────────────────────
    private float[] conv3Filter, conv3Bias;
    private float[] fc1Weight, fc1Bias;  // first FC (6272→128)
    private float[] bnWeight, bnBias, bnMean, bnVar;

    // ── GPU buffers (D3D12 default-heap, UAV) ────────────────────────────
    private MemorySegment inputBuf;
    private MemorySegment conv1FBuf, conv1BBuf, conv1Out;
    private MemorySegment conv2FBuf, conv2BBuf, conv2Out;
    private MemorySegment pool1Out, pool2Out;
    private MemorySegment fcWBuf, fcBBuf, outputBuf;
    // EMNIST-only buffers
    private MemorySegment conv3FBuf, conv3BBuf, conv3Out;
    private MemorySegment fc1WBuf, fc1BBuf, fc1Out;

    // ── Compiled DML operators ───────────────────────────────────────────
    private MemorySegment compiledConv1, compiledPool1;
    private MemorySegment compiledConv2, compiledPool2;
    private MemorySegment compiledGemm;
    // EMNIST-only compiled operators
    private MemorySegment compiledConv3;
    private MemorySegment compiledGemm1;
    private MemorySegment[] allCompiled;

    // ── DML infra ────────────────────────────────────────────────────────
    private MemorySegment descriptorHeap;
    private MemorySegment cmdRecorder;
    private int descriptorIncrement;
    private int totalDescriptors;

    // ── Temp / persistent resources per operator ─────────────────────────
    private long[] tempSizes;
    private long[] persistSizes;
    private int[] descCounts;
    private MemorySegment[] tempBufs;
    private MemorySegment[] persistBufs;

    private boolean loaded = false;
    private boolean closed = false;

    public MnistPipeline(WindowsBindings wb) {
        this.wb = Objects.requireNonNull(wb);
        this.arena = Arena.ofConfined();
    }

    /**
     * Returns the number of output logits (10 for MNIST, 11 for EMNIST).
     */
    public int getOutputSize() {
        return outputSize;
    }

    /**
     * Returns the detected model architecture.
     */
    public ModelArch getArch() {
        return arch;
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Load model
    // ══════════════════════════════════════════════════════════════════════

    public void loadModel(Path onnxFile) throws WindowsNativeException, IOException {
        if (closed) throw new IllegalStateException("MnistPipeline already closed");
        if (loaded)
            throw new IllegalStateException("MnistPipeline already loaded – close and create a new instance to load a different model");
        log.info("MnistPipeline.loadModel({})", onnxFile);
        OnnxModelReader.OnnxGraph graph = OnnxModelReader.parse(onnxFile);

        detectArchitecture(graph);
        extractWeights(graph);

        if (arch == ModelArch.EMNIST_BLANK) {
            createGpuBuffersEmnist();
            uploadWeightsEmnist();
            compileAllOperatorsEmnist();
        } else {
            createGpuBuffers();
            uploadWeights();
            compileAllOperators();
        }

        allocateBindingResources();
        initializeOperators();

        loaded = true;
        log.info("MnistPipeline ready – {} DML operators, arch={}, outputSize={}",
                allCompiled.length, arch, outputSize);
    }

    // ── Architecture detection ───────────────────────────────────────────

    private void detectArchitecture(OnnxModelReader.OnnxGraph graph) {
        long convCount = graph.nodes().stream().filter(n -> "Conv".equals(n.opType())).count();
        boolean hasBatchNorm = graph.nodes().stream().anyMatch(n -> "BatchNormalization".equals(n.opType()));
        boolean isInt8 = graph.nodes().stream().anyMatch(n -> "QLinearConv".equals(n.opType()));

        if (convCount >= 3 && hasBatchNorm && !isInt8) {
            arch = ModelArch.EMNIST_BLANK;
            outputSize = 11;
            log.info("Detected EMNIST_BLANK architecture (3 Conv + BatchNorm, 11 outputs)");
        } else {
            arch = ModelArch.MNIST;
            outputSize = 10;
            log.info("Detected MNIST architecture ({} Conv, {} outputs)", convCount, outputSize);
        }
    }

    // ── Step 1: Extract weights from parsed ONNX graph ───────────────────

    private void extractWeights(OnnxModelReader.OnnxGraph graph) {
        boolean isInt8 = graph.nodes().stream()
                .anyMatch(n -> "QLinearConv".equals(n.opType()));

        if (isInt8) {
            extractWeightsInt8(graph);
        } else if (arch == ModelArch.EMNIST_BLANK) {
            extractWeightsEmnist(graph);
        } else {
            extractWeightsFloat32(graph);
        }

        log.info("Weights extracted for arch={}", arch);
    }

    // ── Float32 weight extraction (mnist-8, mnist-12) ────────────────────

    private void extractWeightsFloat32(OnnxModelReader.OnnxGraph graph) {
        Map<String, OnnxModelReader.OnnxTensor> inits = graph.initializers();

        Map<String, String> reshapeMap = new HashMap<>();
        for (OnnxModelReader.OnnxNode node : graph.nodes()) {
            if ("Reshape".equals(node.opType()) && !node.inputs().isEmpty() && !node.outputs().isEmpty()) {
                reshapeMap.put(node.outputs().get(0), node.inputs().get(0));
            }
        }

        List<float[]> convFilters = new ArrayList<>();
        List<float[]> convBiases = new ArrayList<>();
        float[] matMulWeight = null;
        float[] addBias = null;

        var nodes = graph.nodes();
        for (int i = 0; i < nodes.size(); i++) {
            var node = nodes.get(i);
            switch (node.opType()) {
                case "Conv" -> {
                    OnnxModelReader.OnnxTensor f = resolveInitializer(node.inputs().get(1), inits, reshapeMap);
                    convFilters.add(f != null ? f.data() : new float[0]);

                    float[] bias = null;
                    if (i + 1 < nodes.size() && "Add".equals(nodes.get(i + 1).opType())) {
                        var addNode = nodes.get(i + 1);
                        for (String in : addNode.inputs()) {
                            OnnxModelReader.OnnxTensor t = resolveInitializer(in, inits, reshapeMap);
                            if (t != null) {
                                bias = t.data();
                                break;
                            }
                        }
                    }
                    convBiases.add(bias != null ? bias : new float[0]);
                }
                case "MatMul" -> {
                    OnnxModelReader.OnnxTensor w = resolveInitializer(node.inputs().get(1), inits, reshapeMap);
                    if (w != null) matMulWeight = w.data();
                }
                case "Add" -> {
                    if (matMulWeight != null && addBias == null) {
                        for (String in : node.inputs()) {
                            OnnxModelReader.OnnxTensor t = resolveInitializer(in, inits, reshapeMap);
                            if (t != null) {
                                addBias = t.data();
                                break;
                            }
                        }
                    }
                }
            }
        }

        conv1Filter = convFilters.size() > 0 ? convFilters.get(0) : new float[200];
        conv1Bias = convBiases.size() > 0 ? convBiases.get(0) : new float[8];
        conv2Filter = convFilters.size() > 1 ? convFilters.get(1) : new float[3200];
        conv2Bias = convBiases.size() > 1 ? convBiases.get(1) : new float[16];
        fcWeight = matMulWeight != null ? matMulWeight : new float[2560];
        fcBias = addBias != null ? addBias : new float[10];

        log.info("MNIST weights: conv1F={}, conv2F={}, fcW={}", conv1Filter.length, conv2Filter.length, fcWeight.length);
    }

    // ── EMNIST weight extraction ─────────────────────────────────────────

    /**
     * Extract weights for the EMNIST_BLANK architecture.
     * <p>
     * Graph: Conv(+bias) → Relu → Conv(+bias) → Relu → MaxPool → Conv(+bias) → Relu → MaxPool
     * → Flatten → Gemm(6272→128) → BatchNorm(128) → Relu → Gemm(128→11)
     * <p>
     * Conv nodes have bias as third input (PyTorch export style).
     * Gemm nodes have weight as second input, bias as third.
     * BatchNorm has: scale, bias, mean, var as inputs 1-4.
     */
    private void extractWeightsEmnist(OnnxModelReader.OnnxGraph graph) {
        Map<String, OnnxModelReader.OnnxTensor> inits = graph.initializers();

        // Collect Conv weights: each Conv node has (input, filter, bias)
        List<float[]> convFilters = new ArrayList<>();
        List<float[]> convBiases = new ArrayList<>();

        // Collect Gemm weights
        List<float[]> gemmWeights = new ArrayList<>();
        List<float[]> gemmBiases = new ArrayList<>();

        for (var node : graph.nodes()) {
            switch (node.opType()) {
                case "Conv" -> {
                    var inputs = node.inputs();
                    // inputs[1] = filter, inputs[2] = bias (if present)
                    var f = inits.get(inputs.get(1));
                    convFilters.add(f != null ? f.data() : new float[0]);
                    if (inputs.size() > 2) {
                        var b = inits.get(inputs.get(2));
                        convBiases.add(b != null ? b.data() : new float[0]);
                    } else {
                        convBiases.add(new float[0]);
                    }
                }
                case "Gemm" -> {
                    var inputs = node.inputs();
                    // inputs[1] = weight, inputs[2] = bias
                    var w = inits.get(inputs.get(1));
                    gemmWeights.add(w != null ? w.data() : new float[0]);
                    if (inputs.size() > 2) {
                        var b = inits.get(inputs.get(2));
                        gemmBiases.add(b != null ? b.data() : new float[0]);
                    } else {
                        gemmBiases.add(new float[0]);
                    }
                }
                case "BatchNormalization" -> {
                    var inputs = node.inputs();
                    // inputs: [input, scale, bias, mean, var]
                    var scale = inits.get(inputs.get(1));
                    var bias = inits.get(inputs.get(2));
                    var mean = inits.get(inputs.get(3));
                    var var_ = inits.get(inputs.get(4));
                    bnWeight = scale != null ? scale.data() : new float[128];
                    bnBias = bias != null ? bias.data() : new float[128];
                    bnMean = mean != null ? mean.data() : new float[128];
                    bnVar = var_ != null ? var_.data() : new float[128];
                }
            }
        }

        // 3 Conv layers
        conv1Filter = convFilters.size() > 0 ? convFilters.get(0) : new float[288];
        conv1Bias = convBiases.size() > 0 ? convBiases.get(0) : new float[32];
        conv2Filter = convFilters.size() > 1 ? convFilters.get(1) : new float[18432];
        conv2Bias = convBiases.size() > 1 ? convBiases.get(1) : new float[64];
        conv3Filter = convFilters.size() > 2 ? convFilters.get(2) : new float[73728];
        conv3Bias = convBiases.size() > 2 ? convBiases.get(2) : new float[128];

        // 2 Gemm layers: first = 6272→128, second = 128→11
        fc1Weight = gemmWeights.size() > 0 ? gemmWeights.get(0) : new float[802816];
        fc1Bias = gemmBiases.size() > 0 ? gemmBiases.get(0) : new float[128];
        fcWeight = gemmWeights.size() > 1 ? gemmWeights.get(1) : new float[1408];
        fcBias = gemmBiases.size() > 1 ? gemmBiases.get(1) : new float[11];

        // BN defaults set above in the loop, provide fallback
        if (bnWeight == null) bnWeight = new float[128];
        if (bnBias == null) bnBias = new float[128];
        if (bnMean == null) bnMean = new float[128];
        if (bnVar == null) bnVar = new float[128];

        // ── Inference-mode fusion: fold BatchNorm into fc1 weights/bias ──────
        // BatchNorm is supported via inference-mode fusion, not as a separate
        // general-purpose runtime operator. This is the correct V1 solution.
        // y = scale/sqrt(var+eps) * x + (bias - scale*mean/sqrt(var+eps))
        float eps = 1e-5f;
        int channels = fc1Bias.length;  // 128
        int inputDim = fc1Weight.length / channels;  // 6272
        float[] bnScale = new float[channels];  // scale / sqrt(var + eps)
        float[] bnOffset = new float[channels]; // bias - scale * mean / sqrt(var + eps)
        for (int c = 0; c < channels; c++) {
            bnScale[c] = bnWeight[c] / (float) Math.sqrt(bnVar[c] + eps);
            bnOffset[c] = bnBias[c] - bnScale[c] * bnMean[c];
        }
        // Fuse into fc1: W_fused[c,j] = bnScale[c] * W[c,j], B_fused[c] = bnScale[c] * B[c] + bnOffset[c]
        for (int c = 0; c < channels; c++) {
            for (int j = 0; j < inputDim; j++) {
                fc1Weight[c * inputDim + j] *= bnScale[c];
            }
            fc1Bias[c] = bnScale[c] * fc1Bias[c] + bnOffset[c];
        }
        log.info("BatchNorm folded into fc1 weights/bias");

        log.info("EMNIST weights: conv1F={}, conv2F={}, conv3F={}, fc1W={}, fcW={}, bnFolded=true",
                conv1Filter.length, conv2Filter.length, conv3Filter.length,
                fc1Weight.length, fcWeight.length, bnWeight.length);
    }

    // ── Int8 weight extraction + dequantization (mnist-12-int8) ──────────

    /**
     * Extract quantized weights from an int8 MNIST graph and dequantize to float32.
     */
    private void extractWeightsInt8(OnnxModelReader.OnnxGraph graph) {
        Map<String, OnnxModelReader.OnnxTensor> inits = graph.initializers();

        List<float[]> convFilters = new ArrayList<>();
        List<float[]> convBiases = new ArrayList<>();

        for (var node : graph.nodes()) {
            if ("QLinearConv".equals(node.opType())) {
                var inputs = node.inputs();
                var wQuant = inits.get(inputs.get(3));
                var wScale = inits.get(inputs.get(4));
                var wZp = inits.get(inputs.get(5));

                convFilters.add(dequantizePerChannel(wQuant, wScale, wZp));

                if (inputs.size() > 8) {
                    var xScale = inits.get(inputs.get(1));
                    var biasQuant = inits.get(inputs.get(8));
                    convBiases.add(dequantizeBias(biasQuant, xScale, wScale));
                } else {
                    convBiases.add(new float[0]);
                }
            }
        }

        float[] matMulWeight = null;
        for (var node : graph.nodes()) {
            if ("QLinearMatMul".equals(node.opType())) {
                var inputs = node.inputs();
                var bQuant = inits.get(inputs.get(3));
                var bScale = inits.get(inputs.get(4));
                var bZp = inits.get(inputs.get(5));
                matMulWeight = dequantizePerChannel(bQuant, bScale, bZp);
                break;
            }
        }

        float[] fcBiasResult = null;
        for (var node : graph.nodes()) {
            if ("QLinearAdd".equals(node.opType())) {
                var inputs = node.inputs();
                var bQuant = inits.get(inputs.get(3));
                var bScale = inits.get(inputs.get(4));
                var bZp = inits.get(inputs.get(5));
                fcBiasResult = dequantizeFlat(bQuant, bScale, bZp);
                break;
            }
        }

        conv1Filter = convFilters.size() > 0 ? convFilters.get(0) : new float[200];
        conv1Bias = convBiases.size() > 0 ? convBiases.get(0) : new float[8];
        conv2Filter = convFilters.size() > 1 ? convFilters.get(1) : new float[3200];
        conv2Bias = convBiases.size() > 1 ? convBiases.get(1) : new float[16];
        fcWeight = matMulWeight != null ? matMulWeight : new float[2560];
        fcBias = fcBiasResult != null ? fcBiasResult : new float[10];

        log.info("Int8 MNIST weights: conv1F={}, conv2F={}, fcW={}", conv1Filter.length, conv2Filter.length, fcWeight.length);
    }

    private static float[] dequantizePerChannel(OnnxModelReader.OnnxTensor quant,
                                                OnnxModelReader.OnnxTensor scale,
                                                OnnxModelReader.OnnxTensor zp) {
        if (quant == null || scale == null) return new float[0];
        int totalElements = quant.elementCount();
        int numChannels = scale.data().length;
        int channelStride = numChannels > 0 ? totalElements / numChannels : totalElements;
        float[] result = new float[totalElements];
        byte[] raw = quant.rawBytes();
        for (int i = 0; i < totalElements; i++) {
            int channel = i / channelStride;
            if (channel >= numChannels) channel = numChannels - 1;
            float s = scale.getFloat(channel);
            int zpVal = (zp != null && zp.rawBytes().length > channel) ? zp.getInt8(channel) : 0;
            int qVal = (quant.dataType() == OnnxModelReader.ONNX_UINT8)
                    ? (raw[i] & 0xFF) : raw[i];
            result[i] = s * (qVal - zpVal);
        }
        return result;
    }

    private static float[] dequantizeFlat(OnnxModelReader.OnnxTensor quant,
                                          OnnxModelReader.OnnxTensor scale,
                                          OnnxModelReader.OnnxTensor zp) {
        if (quant == null || scale == null) return new float[0];
        int totalElements = quant.elementCount();
        float s = scale.getFloat(0);
        int zpVal = (zp != null && zp.rawBytes().length > 0) ? (zp.rawBytes()[0] & 0xFF) : 0;
        float[] result = new float[totalElements];
        byte[] raw = quant.rawBytes();
        for (int i = 0; i < totalElements && i < raw.length; i++) {
            int qVal = (quant.dataType() == OnnxModelReader.ONNX_UINT8)
                    ? (raw[i] & 0xFF) : raw[i];
            result[i] = s * (qVal - zpVal);
        }
        return result;
    }

    private static float[] dequantizeBias(OnnxModelReader.OnnxTensor biasQuant,
                                          OnnxModelReader.OnnxTensor xScale,
                                          OnnxModelReader.OnnxTensor wScale) {
        if (biasQuant == null || xScale == null || wScale == null) return new float[0];
        float xS = xScale.getFloat(0);
        int count = biasQuant.elementCount();
        float[] result = new float[count];
        for (int c = 0; c < count; c++) {
            int biasInt = biasQuant.getInt32(c);
            float wS = wScale.getFloat(Math.min(c, wScale.data().length - 1));
            result[c] = biasInt * xS * wS;
        }
        return result;
    }

    // ══════════════════════════════════════════════════════════════════════
    //  MNIST path: GPU buffers, upload, compile, infer
    // ══════════════════════════════════════════════════════════════════════

    private void createGpuBuffers() throws WindowsNativeException {
        var dev = wb.getD3d12Device();
        inputBuf = D3D12Bindings.createDefaultBuffer(dev, fb(784), arena);
        conv1FBuf = D3D12Bindings.createDefaultBuffer(dev, fb(conv1Filter.length), arena);
        conv1BBuf = D3D12Bindings.createDefaultBuffer(dev, fb(conv1Bias.length), arena);
        conv1Out = D3D12Bindings.createDefaultBuffer(dev, fb(6272), arena);
        pool1Out = D3D12Bindings.createDefaultBuffer(dev, fb(1568), arena);
        conv2FBuf = D3D12Bindings.createDefaultBuffer(dev, fb(conv2Filter.length), arena);
        conv2BBuf = D3D12Bindings.createDefaultBuffer(dev, fb(conv2Bias.length), arena);
        conv2Out = D3D12Bindings.createDefaultBuffer(dev, fb(3136), arena);
        pool2Out = D3D12Bindings.createDefaultBuffer(dev, fb(256), arena);
        fcWBuf = D3D12Bindings.createDefaultBuffer(dev, fb(fcWeight.length), arena);
        fcBBuf = D3D12Bindings.createDefaultBuffer(dev, fb(fcBias.length), arena);
        outputBuf = D3D12Bindings.createDefaultBuffer(dev, fb(10), arena);
        log.debug("MNIST GPU buffers created (12 buffers)");
    }

    private void uploadWeights() throws WindowsNativeException {
        var dev = wb.getD3d12Device();
        var q = wb.getCommandQueue();
        D3D12Bindings.uploadFloats(dev, q, conv1FBuf, conv1Filter, arena);
        D3D12Bindings.uploadFloats(dev, q, conv1BBuf, conv1Bias, arena);
        D3D12Bindings.uploadFloats(dev, q, conv2FBuf, conv2Filter, arena);
        D3D12Bindings.uploadFloats(dev, q, conv2BBuf, conv2Bias, arena);
        D3D12Bindings.uploadFloats(dev, q, fcWBuf, fcWeight, arena);
        D3D12Bindings.uploadFloats(dev, q, fcBBuf, fcBias, arena);
        log.debug("MNIST weights uploaded");
    }

    private void compileAllOperators() throws WindowsNativeException {
        compiledConv1 = createAndCompileConv(
                new int[]{1, 1, 28, 28}, new int[]{8, 1, 5, 5}, new int[]{1, 8, 1, 1},
                new int[]{1, 8, 28, 28}, new int[]{1, 1}, new int[]{2, 2}, new int[]{2, 2}, true, true);
        compiledPool1 = createAndCompilePool(
                new int[]{1, 8, 28, 28}, new int[]{1, 8, 14, 14},
                new int[]{2, 2}, new int[]{2, 2});
        compiledConv2 = createAndCompileConv(
                new int[]{1, 8, 14, 14}, new int[]{16, 8, 5, 5}, new int[]{1, 16, 1, 1},
                new int[]{1, 16, 14, 14}, new int[]{1, 1}, new int[]{2, 2}, new int[]{2, 2}, true, true);
        compiledPool2 = createAndCompilePool(
                new int[]{1, 16, 14, 14}, new int[]{1, 16, 4, 4},
                new int[]{3, 3}, new int[]{3, 3});
        compiledGemm = createAndCompileGemm(
                new int[]{1, 1, 1, 256}, new int[]{1, 1, 256, 10},
                new int[]{1, 1, 1, 10}, new int[]{1, 1, 1, 10},
                DirectMlBindings.DML_MATRIX_TRANSFORM_NONE, DirectMlBindings.DML_MATRIX_TRANSFORM_NONE, null);

        allCompiled = new MemorySegment[]{compiledConv1, compiledPool1, compiledConv2, compiledPool2, compiledGemm};
        log.info("MNIST: all {} DML operators compiled", allCompiled.length);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  EMNIST path: GPU buffers, upload, compile
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Create GPU buffers for the EMNIST_BLANK architecture.
     * <p>
     * Tensor sizes:
     * <pre>
     * Input:          (1,1,28,28)   = 784
     * Conv1 out:      (1,32,28,28)  = 25088
     * Conv2 out:      (1,64,28,28)  = 50176
     * Pool1 out:      (1,64,14,14)  = 12544
     * Conv3 out:      (1,128,14,14) = 25088
     * Pool2 out:      (1,128,7,7)   = 6272
     * Gemm1 out (fc1):(1,1,1,128)   = 128
     * BN out:         (1,1,1,128)   = 128
     * Relu out:       (1,1,1,128)   = 128
     * Output (fc2):   (1,1,1,11)    = 11
     * </pre>
     */
    private void createGpuBuffersEmnist() throws WindowsNativeException {
        var dev = wb.getD3d12Device();
        inputBuf = D3D12Bindings.createDefaultBuffer(dev, fb(784), arena);

        // Conv1: (32,1,3,3) filter, 32 bias
        conv1FBuf = D3D12Bindings.createDefaultBuffer(dev, fb(conv1Filter.length), arena);
        conv1BBuf = D3D12Bindings.createDefaultBuffer(dev, fb(conv1Bias.length), arena);
        conv1Out = D3D12Bindings.createDefaultBuffer(dev, fb(25088), arena); // (1,32,28,28)

        // Conv2: (64,32,3,3) filter, 64 bias
        conv2FBuf = D3D12Bindings.createDefaultBuffer(dev, fb(conv2Filter.length), arena);
        conv2BBuf = D3D12Bindings.createDefaultBuffer(dev, fb(conv2Bias.length), arena);
        conv2Out = D3D12Bindings.createDefaultBuffer(dev, fb(50176), arena); // (1,64,28,28)

        // Pool1: (1,64,14,14)
        pool1Out = D3D12Bindings.createDefaultBuffer(dev, fb(12544), arena);

        // Conv3: (128,64,3,3) filter, 128 bias
        conv3FBuf = D3D12Bindings.createDefaultBuffer(dev, fb(conv3Filter.length), arena);
        conv3BBuf = D3D12Bindings.createDefaultBuffer(dev, fb(conv3Bias.length), arena);
        conv3Out = D3D12Bindings.createDefaultBuffer(dev, fb(25088), arena); // (1,128,14,14)

        // Pool2: (1,128,7,7)
        pool2Out = D3D12Bindings.createDefaultBuffer(dev, fb(6272), arena);

        // Gemm1: 6272→128 (with folded BN)
        fc1WBuf = D3D12Bindings.createDefaultBuffer(dev, fb(fc1Weight.length), arena);
        fc1BBuf = D3D12Bindings.createDefaultBuffer(dev, fb(fc1Bias.length), arena);
        fc1Out = D3D12Bindings.createDefaultBuffer(dev, fb(128), arena);


        // Gemm2: 128→11
        fcWBuf = D3D12Bindings.createDefaultBuffer(dev, fb(fcWeight.length), arena);
        fcBBuf = D3D12Bindings.createDefaultBuffer(dev, fb(fcBias.length), arena);
        outputBuf = D3D12Bindings.createDefaultBuffer(dev, fb(11), arena);

        log.debug("EMNIST GPU buffers created");
    }

    private void uploadWeightsEmnist() throws WindowsNativeException {
        var dev = wb.getD3d12Device();
        var q = wb.getCommandQueue();
        D3D12Bindings.uploadFloats(dev, q, conv1FBuf, conv1Filter, arena);
        D3D12Bindings.uploadFloats(dev, q, conv1BBuf, conv1Bias, arena);
        D3D12Bindings.uploadFloats(dev, q, conv2FBuf, conv2Filter, arena);
        D3D12Bindings.uploadFloats(dev, q, conv2BBuf, conv2Bias, arena);
        D3D12Bindings.uploadFloats(dev, q, conv3FBuf, conv3Filter, arena);
        D3D12Bindings.uploadFloats(dev, q, conv3BBuf, conv3Bias, arena);
        D3D12Bindings.uploadFloats(dev, q, fc1WBuf, fc1Weight, arena);  // already includes folded BN
        D3D12Bindings.uploadFloats(dev, q, fc1BBuf, fc1Bias, arena);    // already includes folded BN
        D3D12Bindings.uploadFloats(dev, q, fcWBuf, fcWeight, arena);
        D3D12Bindings.uploadFloats(dev, q, fcBBuf, fcBias, arena);
        log.debug("EMNIST weights uploaded (BN folded into fc1)");
    }

    /**
     * Compile DML operators for EMNIST_BLANK architecture.
     * <p>
     * 7 operators: Conv1+ReLU, Conv2+ReLU, Pool1, Conv3+ReLU, Pool2, Gemm1+ReLU(BN folded), Gemm2
     * <p>
     * BatchNormalization is folded into Gemm1 weights/bias during weight extraction,
     * so no separate BN operator is needed. ReLU after BN is fused into Gemm1.
     */
    private void compileAllOperatorsEmnist() throws WindowsNativeException {
        // Conv1+Relu: (1,1,28,28) → (1,32,28,28), filter(32,1,3,3), bias(1,32,1,1), pad=1, stride=1
        compiledConv1 = createAndCompileConv(
                new int[]{1, 1, 28, 28}, new int[]{32, 1, 3, 3}, new int[]{1, 32, 1, 1},
                new int[]{1, 32, 28, 28}, new int[]{1, 1}, new int[]{1, 1}, new int[]{1, 1}, true, true);

        // Conv2+Relu: (1,32,28,28) → (1,64,28,28), filter(64,32,3,3), bias(1,64,1,1), pad=1, stride=1
        compiledConv2 = createAndCompileConv(
                new int[]{1, 32, 28, 28}, new int[]{64, 32, 3, 3}, new int[]{1, 64, 1, 1},
                new int[]{1, 64, 28, 28}, new int[]{1, 1}, new int[]{1, 1}, new int[]{1, 1}, true, true);

        // Pool1: (1,64,28,28) → (1,64,14,14), window=2, stride=2
        compiledPool1 = createAndCompilePool(
                new int[]{1, 64, 28, 28}, new int[]{1, 64, 14, 14},
                new int[]{2, 2}, new int[]{2, 2});

        // Conv3+Relu: (1,64,14,14) → (1,128,14,14), filter(128,64,3,3), bias(1,128,1,1), pad=1, stride=1
        compiledConv3 = createAndCompileConv(
                new int[]{1, 64, 14, 14}, new int[]{128, 64, 3, 3}, new int[]{1, 128, 1, 1},
                new int[]{1, 128, 14, 14}, new int[]{1, 1}, new int[]{1, 1}, new int[]{1, 1}, true, true);

        // Pool2: (1,128,14,14) → (1,128,7,7), window=2, stride=2
        compiledPool2 = createAndCompilePool(
                new int[]{1, 128, 14, 14}, new int[]{1, 128, 7, 7},
                new int[]{2, 2}, new int[]{2, 2});

        // Gemm1+ReLU: Flatten(6272) → 128, weight is [128,6272] → transB
        // BatchNorm is already folded into fc1 weights/bias, ReLU is fused into Gemm operator
        // A(1,1,1,6272) × B^T(1,1,128,6272) + C(1,1,1,128) → (1,1,1,128)
        MemorySegment reluDesc = arena.allocate(16, 8);
        MemorySegment fusedRelu = DirectMlBindings.allocOperatorDesc(arena,
                DirectMlBindings.DML_OPERATOR_ACTIVATION_RELU, reluDesc);
        compiledGemm1 = createAndCompileGemm(
                new int[]{1, 1, 1, 6272}, new int[]{1, 1, 128, 6272},
                new int[]{1, 1, 1, 128}, new int[]{1, 1, 1, 128},
                DirectMlBindings.DML_MATRIX_TRANSFORM_NONE, DirectMlBindings.DML_MATRIX_TRANSFORM_TRANSPOSE, fusedRelu);

        // Gemm2: 128 → 11, weight is [11,128] → transB
        // A(1,1,1,128) × B^T(1,1,11,128) + C(1,1,1,11) → (1,1,1,11)
        compiledGemm = createAndCompileGemm(
                new int[]{1, 1, 1, 128}, new int[]{1, 1, 11, 128},
                new int[]{1, 1, 1, 11}, new int[]{1, 1, 1, 11},
                DirectMlBindings.DML_MATRIX_TRANSFORM_NONE, DirectMlBindings.DML_MATRIX_TRANSFORM_TRANSPOSE, null);

        allCompiled = new MemorySegment[]{
                compiledConv1, compiledConv2, compiledPool1, compiledConv3, compiledPool2,
                compiledGemm1, compiledGemm
        };
        log.info("EMNIST: all {} DML operators compiled (BN folded, ReLU fused into Gemm1)", allCompiled.length);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Common: binding resources & operator initialization
    // ══════════════════════════════════════════════════════════════════════

    private void allocateBindingResources() throws WindowsNativeException {
        var dml = wb.getDmlDevice();
        var dev = wb.getD3d12Device();
        int opCount = allCompiled.length;

        tempSizes = new long[opCount];
        persistSizes = new long[opCount];
        descCounts = new int[opCount];
        tempBufs = new MemorySegment[opCount];
        persistBufs = new MemorySegment[opCount];
        totalDescriptors = 0;

        for (int i = 0; i < opCount; i++) {
            long[] bp = DirectMlBindings.getBindingProperties(allCompiled[i], arena);
            descCounts[i] = (int) bp[0];
            tempSizes[i] = bp[1];
            persistSizes[i] = bp[2];
            totalDescriptors += Math.max(descCounts[i], 1);

            if (tempSizes[i] > 0) {
                tempBufs[i] = D3D12Bindings.createDefaultBuffer(dev, tempSizes[i], arena);
            }
            if (persistSizes[i] > 0) {
                persistBufs[i] = D3D12Bindings.createDefaultBuffer(dev, persistSizes[i], arena);
            }
            log.debug("Op[{}]: desc={}, temp={}, persist={}", i, descCounts[i], tempSizes[i], persistSizes[i]);
        }

        totalDescriptors = Math.max(totalDescriptors, 32) + 32;
        descriptorHeap = D3D12Bindings.createDescriptorHeap(dev, totalDescriptors, arena);
        descriptorIncrement = D3D12Bindings.getDescriptorIncrementSize(dev);
        cmdRecorder = DirectMlBindings.createCommandRecorder(dml, arena);

        log.debug("Descriptor heap: {} descriptors, increment={}", totalDescriptors, descriptorIncrement);
    }

    private void initializeOperators() throws WindowsNativeException {
        boolean anyPersist = false;
        for (long ps : persistSizes)
            if (ps > 0) {
                anyPersist = true;
                break;
            }
        if (!anyPersist) {
            log.info("No persistent resources needed – skipping operator initialization");
            return;
        }

        var dml = wb.getDmlDevice();
        var dev = wb.getD3d12Device();
        var q = wb.getCommandQueue();
        int opCount = allCompiled.length;

        MemorySegment initializer = DirectMlBindings.createOperatorInitializer(dml, allCompiled, arena);
        long[] initBp = DirectMlBindings.getBindingProperties(initializer, arena);
        int initDescCount = Math.max((int) initBp[0], 1);
        long initTempSize = initBp[1];

        long cpuStart = D3D12Bindings.getCpuDescriptorHandleForHeapStart(descriptorHeap, arena);
        long gpuStart = D3D12Bindings.getGpuDescriptorHandleForHeapStart(descriptorHeap, arena);

        MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(arena, initializer,
                cpuStart, gpuStart, initDescCount);
        MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, arena);

        MemorySegment inputBindings = arena.allocate(16L * opCount, 8);
        for (int i = 0; i < opCount; i++) {
            setBindingDesc(inputBindings, i, DirectMlBindings.DML_BINDING_TYPE_NONE, MemorySegment.NULL);
        }
        DirectMlBindings.bindInputs(bt, opCount, inputBindings);

        if (initTempSize > 0) {
            MemorySegment initTmp = D3D12Bindings.createDefaultBuffer(dev, initTempSize, arena);
            MemorySegment tmpBb = DirectMlBindings.allocBufferBinding(arena, initTmp, 0, initTempSize);
            MemorySegment tmpBd = DirectMlBindings.allocBindingDesc(arena, DirectMlBindings.DML_BINDING_TYPE_BUFFER, tmpBb);
            DirectMlBindings.bindTemporaryResource(bt, tmpBd);
        }

        var alloc = D3D12Bindings.createCommandAllocator(dev, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        MemorySegment cmdList = null;
        try {
            cmdList = D3D12Bindings.createCommandList(dev, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, arena);
            D3D12Bindings.setDescriptorHeaps(cmdList, descriptorHeap, arena);
            DirectMlBindings.recordDispatch(cmdRecorder, cmdList, initializer, bt);
            D3D12Bindings.executeAndWait(dev, q, cmdList, arena);
        } finally {
            if (cmdList != null) DxgiBindings.release(cmdList);
            DxgiBindings.release(alloc);
            DxgiBindings.release(bt);
            DxgiBindings.release(initializer);
        }
        log.info("Operator initialization complete");
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Inference
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Run inference. Input: 784 floats (28×28 grayscale, 0.0–1.0).
     * Output: 10 logits (MNIST) or 11 logits (EMNIST).
     */
    public float[] infer(float[] input) throws WindowsNativeException {
        if (closed) throw new WindowsNativeException("Pipeline closed – cannot infer");
        if (!loaded) throw new WindowsNativeException("Pipeline not loaded – call loadModel() first");
        if (input.length != 784) throw new WindowsNativeException("Expected 784 floats, got " + input.length);

        return (arch == ModelArch.EMNIST_BLANK) ? inferEmnist(input) : inferMnist(input);
    }

    private float[] inferMnist(float[] input) throws WindowsNativeException {
        var dev = wb.getD3d12Device();
        var q = wb.getCommandQueue();
        var dml = wb.getDmlDevice();

        D3D12Bindings.uploadFloats(dev, q, inputBuf, input, arena);

        var alloc = D3D12Bindings.createCommandAllocator(dev, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        MemorySegment cmdList = null;
        try {
            cmdList = D3D12Bindings.createCommandList(dev, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, arena);
            D3D12Bindings.setDescriptorHeaps(cmdList, descriptorHeap, arena);

            long cpuBase = D3D12Bindings.getCpuDescriptorHandleForHeapStart(descriptorHeap, arena);
            long gpuBase = D3D12Bindings.getGpuDescriptorHandleForHeapStart(descriptorHeap, arena);
            int descOffset = 0;

            // Op 0: Conv1 + Relu
            descOffset = dispatchConv(dml, cmdList, compiledConv1, 0,
                    inputBuf, fb(784), conv1FBuf, fb(conv1Filter.length),
                    conv1BBuf, fb(conv1Bias.length), conv1Out, fb(6272),
                    true, cpuBase, gpuBase, descOffset);
            D3D12Bindings.uavBarrier(cmdList, arena);

            // Op 1: MaxPool1
            descOffset = dispatchPool(dml, cmdList, compiledPool1, 1,
                    conv1Out, fb(6272), pool1Out, fb(1568),
                    cpuBase, gpuBase, descOffset);
            D3D12Bindings.uavBarrier(cmdList, arena);

            // Op 2: Conv2 + Relu
            descOffset = dispatchConv(dml, cmdList, compiledConv2, 2,
                    pool1Out, fb(1568), conv2FBuf, fb(conv2Filter.length),
                    conv2BBuf, fb(conv2Bias.length), conv2Out, fb(3136),
                    true, cpuBase, gpuBase, descOffset);
            D3D12Bindings.uavBarrier(cmdList, arena);

            // Op 3: MaxPool2
            descOffset = dispatchPool(dml, cmdList, compiledPool2, 3,
                    conv2Out, fb(3136), pool2Out, fb(256),
                    cpuBase, gpuBase, descOffset);
            D3D12Bindings.uavBarrier(cmdList, arena);

            // Op 4: Gemm
            descOffset = dispatchGemm(dml, cmdList, compiledGemm, 4,
                    pool2Out, fb(256), fcWBuf, fb(fcWeight.length),
                    fcBBuf, fb(fcBias.length), outputBuf, fb(10),
                    cpuBase, gpuBase, descOffset);

            D3D12Bindings.executeAndWait(dev, q, cmdList, arena);
            float[] result = D3D12Bindings.readbackFloats(dev, q, outputBuf, 10, arena);
            log.debug("MNIST inference complete: {}", Arrays.toString(result));
            return result;

        } finally {
            if (cmdList != null) DxgiBindings.release(cmdList);
            DxgiBindings.release(alloc);
        }
    }

    private float[] inferEmnist(float[] input) throws WindowsNativeException {
        var dev = wb.getD3d12Device();
        var q = wb.getCommandQueue();
        var dml = wb.getDmlDevice();

        D3D12Bindings.uploadFloats(dev, q, inputBuf, input, arena);

        long cpuBase = D3D12Bindings.getCpuDescriptorHandleForHeapStart(descriptorHeap, arena);
        long gpuBase = D3D12Bindings.getGpuDescriptorHandleForHeapStart(descriptorHeap, arena);
        int descOffset = 0;

        // Op 0: Conv1+Relu (1,1,28,28)→(1,32,28,28)
        descOffset = dispatchSingleOp(dev, q, dml, compiledConv1, 0, cpuBase, gpuBase, descOffset,
                bt -> {
                    MemorySegment inputs2 = arena.allocate(16L * 3, 8);
                    setBufferBinding(inputs2, 0, inputBuf, fb(784));
                    setBufferBinding(inputs2, 1, conv1FBuf, fb(conv1Filter.length));
                    setBufferBinding(inputs2, 2, conv1BBuf, fb(conv1Bias.length));
                    DirectMlBindings.bindInputs(bt, 3, inputs2);
                    MemorySegment outputs = arena.allocate(16, 8);
                    setBufferBinding(outputs, 0, conv1Out, fb(25088));
                    DirectMlBindings.bindOutputs(bt, 1, outputs);
                });
        log.debug("Op0 (Conv1) done");

        // Op 1: Conv2+Relu (1,32,28,28)→(1,64,28,28)
        descOffset = dispatchSingleOp(dev, q, dml, compiledConv2, 1, cpuBase, gpuBase, descOffset,
                bt -> {
                    MemorySegment inputs2 = arena.allocate(16L * 3, 8);
                    setBufferBinding(inputs2, 0, conv1Out, fb(25088));
                    setBufferBinding(inputs2, 1, conv2FBuf, fb(conv2Filter.length));
                    setBufferBinding(inputs2, 2, conv2BBuf, fb(conv2Bias.length));
                    DirectMlBindings.bindInputs(bt, 3, inputs2);
                    MemorySegment outputs = arena.allocate(16, 8);
                    setBufferBinding(outputs, 0, conv2Out, fb(50176));
                    DirectMlBindings.bindOutputs(bt, 1, outputs);
                });
        log.debug("Op1 (Conv2) done");

        // Op 2: Pool1 (1,64,28,28)→(1,64,14,14)
        descOffset = dispatchSingleOp(dev, q, dml, compiledPool1, 2, cpuBase, gpuBase, descOffset,
                bt -> {
                    MemorySegment inputs2 = arena.allocate(16, 8);
                    setBufferBinding(inputs2, 0, conv2Out, fb(50176));
                    DirectMlBindings.bindInputs(bt, 1, inputs2);
                    MemorySegment outputs = arena.allocate(16, 8);
                    setBufferBinding(outputs, 0, pool1Out, fb(12544));
                    DirectMlBindings.bindOutputs(bt, 1, outputs);
                });
        log.debug("Op2 (Pool1) done");

        // Op 3: Conv3+Relu (1,64,14,14)→(1,128,14,14)
        descOffset = dispatchSingleOp(dev, q, dml, compiledConv3, 3, cpuBase, gpuBase, descOffset,
                bt -> {
                    MemorySegment inputs2 = arena.allocate(16L * 3, 8);
                    setBufferBinding(inputs2, 0, pool1Out, fb(12544));
                    setBufferBinding(inputs2, 1, conv3FBuf, fb(conv3Filter.length));
                    setBufferBinding(inputs2, 2, conv3BBuf, fb(conv3Bias.length));
                    DirectMlBindings.bindInputs(bt, 3, inputs2);
                    MemorySegment outputs = arena.allocate(16, 8);
                    setBufferBinding(outputs, 0, conv3Out, fb(25088));
                    DirectMlBindings.bindOutputs(bt, 1, outputs);
                });
        log.debug("Op3 (Conv3) done");

        // Op 4: Pool2 (1,128,14,14)→(1,128,7,7)
        descOffset = dispatchSingleOp(dev, q, dml, compiledPool2, 4, cpuBase, gpuBase, descOffset,
                bt -> {
                    MemorySegment inputs2 = arena.allocate(16, 8);
                    setBufferBinding(inputs2, 0, conv3Out, fb(25088));
                    DirectMlBindings.bindInputs(bt, 1, inputs2);
                    MemorySegment outputs = arena.allocate(16, 8);
                    setBufferBinding(outputs, 0, pool2Out, fb(6272));
                    DirectMlBindings.bindOutputs(bt, 1, outputs);
                });
        log.debug("Op4 (Pool2) done");

        // Op 5: Gemm1+ReLU (6272→128, BN folded into weights, ReLU fused)
        descOffset = dispatchSingleOp(dev, q, dml, compiledGemm1, 5, cpuBase, gpuBase, descOffset,
                bt -> {
                    MemorySegment inputs2 = arena.allocate(16L * 3, 8);
                    setBufferBinding(inputs2, 0, pool2Out, fb(6272));
                    setBufferBinding(inputs2, 1, fc1WBuf, fb(fc1Weight.length));
                    setBufferBinding(inputs2, 2, fc1BBuf, fb(fc1Bias.length));
                    DirectMlBindings.bindInputs(bt, 3, inputs2);
                    MemorySegment outputs = arena.allocate(16, 8);
                    setBufferBinding(outputs, 0, fc1Out, fb(128));
                    DirectMlBindings.bindOutputs(bt, 1, outputs);
                });
        log.debug("Op5 (Gemm1+ReLU, BN folded) done");

        // Op 6: Gemm2 (128→11)
        descOffset = dispatchSingleOp(dev, q, dml, compiledGemm, 6, cpuBase, gpuBase, descOffset,
                bt -> {
                    MemorySegment inputs2 = arena.allocate(16L * 3, 8);
                    setBufferBinding(inputs2, 0, fc1Out, fb(128));
                    setBufferBinding(inputs2, 1, fcWBuf, fb(fcWeight.length));
                    setBufferBinding(inputs2, 2, fcBBuf, fb(fcBias.length));
                    DirectMlBindings.bindInputs(bt, 3, inputs2);
                    MemorySegment outputs = arena.allocate(16, 8);
                    setBufferBinding(outputs, 0, outputBuf, fb(11));
                    DirectMlBindings.bindOutputs(bt, 1, outputs);
                });
        log.debug("Op6 (Gemm2) done");

        float[] result = D3D12Bindings.readbackFloats(dev, q, outputBuf, outputSize, arena);
        log.debug("EMNIST inference complete: {}", Arrays.toString(result));
        return result;
    }

    /**
     * Functional interface for binding a single operator's inputs/outputs.
     */
    @FunctionalInterface
    private interface BindingAction {
        void bind(MemorySegment bindingTable);
    }

    /**
     * Dispatch and execute a single DML operator (record → execute → wait).
     * This per-op execution strategy isolates GPU faults to the exact failing operator.
     */
    private int dispatchSingleOp(MemorySegment dev, MemorySegment q, MemorySegment dml,
                                 MemorySegment compiled, int opIdx,
                                 long cpuBase, long gpuBase, int descOff,
                                 BindingAction bindAction) throws WindowsNativeException {
        int descCount = Math.max(descCounts[opIdx], 1);

        MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(arena, compiled,
                cpuBase + (long) descOff * descriptorIncrement,
                gpuBase + (long) descOff * descriptorIncrement, descCount);
        MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, arena);

        bindAction.bind(bt);
        bindTempAndPersist(bt, opIdx);

        var alloc = D3D12Bindings.createCommandAllocator(dev, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        MemorySegment cmdList = null;
        try {
            cmdList = D3D12Bindings.createCommandList(dev, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, arena);
            D3D12Bindings.setDescriptorHeaps(cmdList, descriptorHeap, arena);
            DirectMlBindings.recordDispatch(cmdRecorder, cmdList, compiled, bt);
            D3D12Bindings.executeAndWait(dev, q, cmdList, arena);
        } finally {
            if (cmdList != null) DxgiBindings.release(cmdList);
            DxgiBindings.release(alloc);
            DxgiBindings.release(bt);
        }

        return descOff + descCount;
    }

    /**
     * Compute argmax of output logits.
     */
    public static int argmax(float[] arr) {
        int idx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[idx]) idx = i;
        }
        return idx;
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Dispatch helpers
    // ══════════════════════════════════════════════════════════════════════

    private int dispatchConv(MemorySegment dml, MemorySegment cmdList,
                             MemorySegment compiled, int opIdx,
                             MemorySegment inBuf, long inSize,
                             MemorySegment filterBuf, long filterSize,
                             MemorySegment biasBuf, long biasSize,
                             MemorySegment outBuf, long outSize,
                             boolean hasBias,
                             long cpuBase, long gpuBase, int descOff)
            throws WindowsNativeException {
        int descCount = Math.max(descCounts[opIdx], 1);

        MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(arena, compiled,
                cpuBase + (long) descOff * descriptorIncrement,
                gpuBase + (long) descOff * descriptorIncrement, descCount);
        MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, arena);

        if (hasBias) {
            MemorySegment inputs = arena.allocate(16L * 3, 8);
            setBufferBinding(inputs, 0, inBuf, inSize);
            setBufferBinding(inputs, 1, filterBuf, filterSize);
            setBufferBinding(inputs, 2, biasBuf, biasSize);
            DirectMlBindings.bindInputs(bt, 3, inputs);
        } else {
            MemorySegment inputs = arena.allocate(16L * 2, 8);
            setBufferBinding(inputs, 0, inBuf, inSize);
            setBufferBinding(inputs, 1, filterBuf, filterSize);
            DirectMlBindings.bindInputs(bt, 2, inputs);
        }

        MemorySegment outputs = arena.allocate(16, 8);
        setBufferBinding(outputs, 0, outBuf, outSize);
        DirectMlBindings.bindOutputs(bt, 1, outputs);

        bindTempAndPersist(bt, opIdx);
        DirectMlBindings.recordDispatch(cmdRecorder, cmdList, compiled, bt);
        DxgiBindings.release(bt);
        return descOff + descCount;
    }

    private int dispatchPool(MemorySegment dml, MemorySegment cmdList,
                             MemorySegment compiled, int opIdx,
                             MemorySegment inBuf, long inSize,
                             MemorySegment outBuf, long outSize,
                             long cpuBase, long gpuBase, int descOff)
            throws WindowsNativeException {
        int descCount = Math.max(descCounts[opIdx], 1);

        MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(arena, compiled,
                cpuBase + (long) descOff * descriptorIncrement,
                gpuBase + (long) descOff * descriptorIncrement, descCount);
        MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, arena);

        MemorySegment inputs = arena.allocate(16, 8);
        setBufferBinding(inputs, 0, inBuf, inSize);
        DirectMlBindings.bindInputs(bt, 1, inputs);

        MemorySegment outputs = arena.allocate(16, 8);
        setBufferBinding(outputs, 0, outBuf, outSize);
        DirectMlBindings.bindOutputs(bt, 1, outputs);

        bindTempAndPersist(bt, opIdx);
        DirectMlBindings.recordDispatch(cmdRecorder, cmdList, compiled, bt);
        DxgiBindings.release(bt);
        return descOff + descCount;
    }

    private int dispatchGemm(MemorySegment dml, MemorySegment cmdList,
                             MemorySegment compiled, int opIdx,
                             MemorySegment aBuf, long aSize,
                             MemorySegment bBuf, long bSize,
                             MemorySegment cBuf, long cSize,
                             MemorySegment outBuf, long outSize,
                             long cpuBase, long gpuBase, int descOff)
            throws WindowsNativeException {
        int descCount = Math.max(descCounts[opIdx], 1);

        MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(arena, compiled,
                cpuBase + (long) descOff * descriptorIncrement,
                gpuBase + (long) descOff * descriptorIncrement, descCount);
        MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, arena);

        MemorySegment inputs = arena.allocate(16L * 3, 8);
        setBufferBinding(inputs, 0, aBuf, aSize);
        setBufferBinding(inputs, 1, bBuf, bSize);
        setBufferBinding(inputs, 2, cBuf, cSize);
        DirectMlBindings.bindInputs(bt, 3, inputs);

        MemorySegment outputs = arena.allocate(16, 8);
        setBufferBinding(outputs, 0, outBuf, outSize);
        DirectMlBindings.bindOutputs(bt, 1, outputs);

        bindTempAndPersist(bt, opIdx);
        DirectMlBindings.recordDispatch(cmdRecorder, cmdList, compiled, bt);
        DxgiBindings.release(bt);
        return descOff + descCount;
    }


    private void bindTempAndPersist(MemorySegment bt, int opIdx) {
        if (tempSizes[opIdx] > 0 && tempBufs[opIdx] != null) {
            MemorySegment bb = DirectMlBindings.allocBufferBinding(arena, tempBufs[opIdx], 0, tempSizes[opIdx]);
            MemorySegment bd = DirectMlBindings.allocBindingDesc(arena, DirectMlBindings.DML_BINDING_TYPE_BUFFER, bb);
            DirectMlBindings.bindTemporaryResource(bt, bd);
        }
        if (persistSizes[opIdx] > 0 && persistBufs[opIdx] != null) {
            MemorySegment bb = DirectMlBindings.allocBufferBinding(arena, persistBufs[opIdx], 0, persistSizes[opIdx]);
            MemorySegment bd = DirectMlBindings.allocBindingDesc(arena, DirectMlBindings.DML_BINDING_TYPE_BUFFER, bb);
            DirectMlBindings.bindPersistentResource(bt, bd);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Operator creation helpers
    // ══════════════════════════════════════════════════════════════════════

    private MemorySegment createAndCompileConv(int[] inShape, int[] filterShape, int[] biasShape,
                                               int[] outShape, int[] strides, int[] padStart,
                                               int[] padEnd, boolean fuseRelu, boolean hasBias)
            throws WindowsNativeException {
        var dml = wb.getDmlDevice();
        var inTD = td(inShape);
        var filterTD = td(filterShape);
        var outTD = td(outShape);

        MemorySegment conv = arena.allocate(104, 8);
        conv.set(ValueLayout.ADDRESS, 0, inTD);
        conv.set(ValueLayout.ADDRESS, 8, filterTD);
        if (hasBias) {
            conv.set(ValueLayout.ADDRESS, 16, td(biasShape));
        } else {
            conv.set(ValueLayout.ADDRESS, 16, MemorySegment.NULL);
        }
        conv.set(ValueLayout.ADDRESS, 24, outTD);
        conv.set(ValueLayout.JAVA_INT, 32, DirectMlBindings.DML_CONVOLUTION_MODE_CROSS_CORRELATION);
        conv.set(ValueLayout.JAVA_INT, 36, DirectMlBindings.DML_CONVOLUTION_DIRECTION_FORWARD);
        conv.set(ValueLayout.JAVA_INT, 40, 2);

        conv.set(ValueLayout.ADDRESS, 48, allocInts(strides));
        conv.set(ValueLayout.ADDRESS, 56, allocInts(new int[]{1, 1}));
        conv.set(ValueLayout.ADDRESS, 64, allocInts(padStart));
        conv.set(ValueLayout.ADDRESS, 72, allocInts(padEnd));
        conv.set(ValueLayout.ADDRESS, 80, allocInts(new int[]{0, 0}));
        conv.set(ValueLayout.JAVA_INT, 88, 1);

        if (fuseRelu) {
            MemorySegment reluDesc = arena.allocate(16, 8);
            MemorySegment fusedAct = DirectMlBindings.allocOperatorDesc(arena,
                    DirectMlBindings.DML_OPERATOR_ACTIVATION_RELU, reluDesc);
            conv.set(ValueLayout.ADDRESS, 96, fusedAct);
        } else {
            conv.set(ValueLayout.ADDRESS, 96, MemorySegment.NULL);
        }

        MemorySegment opDesc = DirectMlBindings.allocOperatorDesc(arena,
                DirectMlBindings.DML_OPERATOR_CONVOLUTION, conv);
        MemorySegment op = DirectMlBindings.createOperator(dml, opDesc, arena);
        MemorySegment compiled = DirectMlBindings.compileOperator(dml, op, DirectMlBindings.DML_EXECUTION_FLAG_NONE, arena);
        DxgiBindings.release(op);
        return compiled;
    }

    private MemorySegment createAndCompilePool(int[] inShape, int[] outShape,
                                               int[] windowSize, int[] strides)
            throws WindowsNativeException {
        var dml = wb.getDmlDevice();
        var inTD = td(inShape);
        var outTD = td(outShape);

        MemorySegment pool = arena.allocate(56, 8);
        pool.set(ValueLayout.ADDRESS, 0, inTD);
        pool.set(ValueLayout.ADDRESS, 8, outTD);
        pool.set(ValueLayout.JAVA_INT, 16, 2);
        pool.set(ValueLayout.ADDRESS, 24, allocInts(strides));
        pool.set(ValueLayout.ADDRESS, 32, allocInts(windowSize));
        pool.set(ValueLayout.ADDRESS, 40, allocInts(new int[]{0, 0}));
        pool.set(ValueLayout.ADDRESS, 48, allocInts(new int[]{0, 0}));

        MemorySegment opDesc = DirectMlBindings.allocOperatorDesc(arena,
                DirectMlBindings.DML_OPERATOR_MAX_POOLING, pool);
        MemorySegment op = DirectMlBindings.createOperator(dml, opDesc, arena);
        MemorySegment compiled = DirectMlBindings.compileOperator(dml, op, DirectMlBindings.DML_EXECUTION_FLAG_NONE, arena);
        DxgiBindings.release(op);
        return compiled;
    }

    /**
     * Create and compile a DML Gemm operator.
     *
     * @param transA          DML_MATRIX_TRANSFORM for A
     * @param transB          DML_MATRIX_TRANSFORM for B
     * @param fusedActivation optional fused activation (null = none)
     */
    private MemorySegment createAndCompileGemm(int[] aShape, int[] bShape,
                                               int[] cShape, int[] outShape,
                                               int transA, int transB,
                                               MemorySegment fusedActivation)
            throws WindowsNativeException {
        var dml = wb.getDmlDevice();

        MemorySegment gemm = arena.allocate(56, 8);
        gemm.set(ValueLayout.ADDRESS, 0, td(aShape));
        gemm.set(ValueLayout.ADDRESS, 8, td(bShape));
        gemm.set(ValueLayout.ADDRESS, 16, td(cShape));
        gemm.set(ValueLayout.ADDRESS, 24, td(outShape));
        gemm.set(ValueLayout.JAVA_INT, 32, transA);
        gemm.set(ValueLayout.JAVA_INT, 36, transB);
        gemm.set(ValueLayout.JAVA_FLOAT, 40, 1.0f);
        gemm.set(ValueLayout.JAVA_FLOAT, 44, 1.0f);
        gemm.set(ValueLayout.ADDRESS, 48, fusedActivation != null ? fusedActivation : MemorySegment.NULL);

        MemorySegment opDesc = DirectMlBindings.allocOperatorDesc(arena,
                DirectMlBindings.DML_OPERATOR_GEMM, gemm);
        MemorySegment op = DirectMlBindings.createOperator(dml, opDesc, arena);
        MemorySegment compiled = DirectMlBindings.compileOperator(dml, op, DirectMlBindings.DML_EXECUTION_FLAG_NONE, arena);
        DxgiBindings.release(op);
        return compiled;
    }


    // ══════════════════════════════════════════════════════════════════════
    //  Internal helpers
    // ══════════════════════════════════════════════════════════════════════

    private MemorySegment td(int[] sizes) {
        int elems = 1;
        for (int s : sizes) elems *= s;
        long byteSize = (long) elems * Float.BYTES;
        var bufTD = DirectMlBindings.allocBufferTensorDesc(arena,
                DirectMlBindings.DML_TENSOR_DATA_TYPE_FLOAT32, sizes, null, byteSize);
        return DirectMlBindings.allocTensorDesc(arena, bufTD);
    }

    private MemorySegment allocInts(int[] values) {
        MemorySegment seg = arena.allocate((long) values.length * ValueLayout.JAVA_INT.byteSize(), 4);
        for (int i = 0; i < values.length; i++) seg.setAtIndex(ValueLayout.JAVA_INT, i, values[i]);
        return seg;
    }

    private static OnnxModelReader.OnnxTensor resolveInitializer(
            String name, Map<String, OnnxModelReader.OnnxTensor> inits,
            Map<String, String> reshapeMap) {
        OnnxModelReader.OnnxTensor t = inits.get(name);
        if (t != null) return t;
        String traced = name;
        for (int i = 0; i < 5 && traced != null; i++) {
            traced = reshapeMap.get(traced);
            if (traced != null) {
                t = inits.get(traced);
                if (t != null) return t;
            }
        }
        return null;
    }

    private static long fb(int elementCount) {
        return (long) elementCount * Float.BYTES;
    }

    private void setBufferBinding(MemorySegment array, int index, MemorySegment buffer, long size) {
        MemorySegment bb = DirectMlBindings.allocBufferBinding(arena, buffer, 0, size);
        long off = (long) index * 16;
        array.set(ValueLayout.JAVA_INT, off, DirectMlBindings.DML_BINDING_TYPE_BUFFER);
        array.set(ValueLayout.ADDRESS, off + 8, bb);
    }

    private void setBindingDesc(MemorySegment array, int index, int type, MemorySegment desc) {
        long off = (long) index * 16;
        array.set(ValueLayout.JAVA_INT, off, type);
        array.set(ValueLayout.ADDRESS, off + 8, desc);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Cleanup
    // ══════════════════════════════════════════════════════════════════════

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        loaded = false;

        safeRelease(cmdRecorder, "cmdRecorder");
        safeRelease(descriptorHeap, "descriptorHeap");
        if (allCompiled != null) {
            for (var c : allCompiled) safeRelease(c, "compiled operator");
        }
        // Release GPU buffers – shared
        for (var buf : new MemorySegment[]{inputBuf, conv1FBuf, conv1BBuf, conv1Out, pool1Out,
                conv2FBuf, conv2BBuf, conv2Out, pool2Out, fcWBuf, fcBBuf, outputBuf}) {
            safeRelease(buf, "GPU buffer");
        }
        // Release EMNIST-only buffers
        for (var buf : new MemorySegment[]{conv3FBuf, conv3BBuf, conv3Out,
                fc1WBuf, fc1BBuf, fc1Out}) {
            safeRelease(buf, "EMNIST GPU buffer");
        }
        if (tempBufs != null) for (var b : tempBufs) safeRelease(b, "temp buffer");
        if (persistBufs != null) for (var b : persistBufs) safeRelease(b, "persist buffer");
        if (arena != null) arena.close();
        log.info("MnistPipeline closed");
    }

    private static void safeRelease(MemorySegment comPtr, String label) {
        if (comPtr == null || comPtr.equals(MemorySegment.NULL)) return;
        try {
            DxgiBindings.release(comPtr);
        } catch (Exception e) {
            log.warn("Failed to release {}: {}", label, e.getMessage());
        }
    }
}
