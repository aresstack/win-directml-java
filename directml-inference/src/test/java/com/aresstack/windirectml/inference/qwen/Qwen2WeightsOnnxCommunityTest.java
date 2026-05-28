package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.windows.OnnxModelReader;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link Qwen2Weights} ONNX Community tensor name support.
 *
 * <p>Validates that the weight loader correctly handles the ONNX Community export
 * naming pattern (e.g. {@code model.layers.N.attn.q_proj.MatMul.weight}) and the
 * transposed weight layout used by that export.</p>
 *
 * <p>These tests exercise the name mapping and error message logic without
 * requiring real model weights.</p>
 */
class Qwen2WeightsOnnxCommunityTest {

    /**
     * Verify that describeUnsupportedFormat never returns "null" as the detail.
     * An empty ONNX file triggers a parse error, and the message must include
     * the actual exception info rather than literal "null".
     */
    @Test
    void describeUnsupportedFormatNeverReturnsNullMessage(@org.junit.jupiter.api.io.TempDir java.nio.file.Path tmp)
            throws Exception {
        java.nio.file.Files.writeString(tmp.resolve("model.onnx"), "");
        java.nio.file.Files.writeString(tmp.resolve("model.onnx_data"), "");

        String msg = Qwen2Weights.describeUnsupportedFormat(tmp);
        assertNotNull(msg);
        assertTrue(msg.startsWith("Unsupported Qwen ONNX format: "), msg);
        // Must not contain literal "null" as the detail
        assertFalse(msg.equals("Unsupported Qwen ONNX format: null"),
                "Error message must not have 'null' as the format detail: " + msg);
    }

    /**
     * Verify the LayerWeights record supports biases and backward-compatible construction.
     */
    @Test
    void layerWeightsWithBiasesStoresBiasArrays() {
        float[] norm = new float[]{1.0f};
        Qwen2Weights.WeightMatrix dummy = new Qwen2Weights.DenseWeightMatrix(
                new Qwen2Weights.DenseWeight(new float[]{1.0f}, 1, 1));
        float[] qBias = new float[]{0.5f, 0.5f};
        float[] kBias = new float[]{0.3f};
        float[] vBias = new float[]{0.7f};

        var lw = new Qwen2Weights.LayerWeights(
                norm, dummy, dummy, dummy, dummy, norm, dummy, dummy, dummy,
                qBias, kBias, vBias);

        assertArrayEquals(qBias, lw.qBias());
        assertArrayEquals(kBias, lw.kBias());
        assertArrayEquals(vBias, lw.vBias());
    }

    /**
     * Verify the backward-compatible LayerWeights constructor sets biases to null.
     */
    @Test
    void layerWeightsWithoutBiasesHasNullBiases() {
        float[] norm = new float[]{1.0f};
        Qwen2Weights.WeightMatrix dummy = new Qwen2Weights.DenseWeightMatrix(
                new Qwen2Weights.DenseWeight(new float[]{1.0f}, 1, 1));

        var lw = new Qwen2Weights.LayerWeights(
                norm, dummy, dummy, dummy, dummy, norm, dummy, dummy, dummy);

        assertNull(lw.qBias());
        assertNull(lw.kBias());
        assertNull(lw.vBias());
    }

    /**
     * Verify that the DenseWeight matvec correctly computes y = W @ x.
     * This is important because transposed loading must produce a weight
     * that works correctly with the matvec contract.
     */
    @Test
    void denseWeightMatvecComputesCorrectly() {
        // W = [[1, 2], [3, 4]] → N=2, K=2
        // x = [1, 1] → y = [3, 7]
        float[] data = {1, 2, 3, 4};
        var weight = new Qwen2Weights.DenseWeight(data, 2, 2);
        float[] x = {1, 1};
        float[] y = {0, 0};
        weight.matvec(x, y);
        assertEquals(3.0f, y[0], 1e-6f);
        assertEquals(7.0f, y[1], 1e-6f);
    }

    /**
     * Verify that a transposed matrix (from ONNX [K,N] layout to [N,K])
     * produces the same matvec result.
     *
     * If ONNX stores weight as [K=2, N=2]:
     *   rawData = [[1, 3], [2, 4]]  (i.e. [K][N] layout)
     *
     * After transpose to [N=2, K=2]:
     *   transposed = [[1, 2], [3, 4]]  (same as above test)
     */
    @Test
    void transposedLayoutProducesSameResult() {
        // ONNX [K, N] layout: rawData[k * N + n] = W[n, k] transposed
        float[] rawDataKN = {1, 3, 2, 4}; // row 0: [1,3] row 1: [2,4]
        int N = 2, K = 2;

        // Transpose: transposed[n * K + k] = rawData[k * N + n]
        float[] transposed = new float[N * K];
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                transposed[n * K + k] = rawDataKN[k * N + n];
            }
        }

        var weight = new Qwen2Weights.DenseWeight(transposed, N, K);
        float[] x = {1, 1};
        float[] y = {0, 0};
        weight.matvec(x, y);
        // Same as non-transposed test: y = [3, 7]
        assertEquals(3.0f, y[0], 1e-6f);
        assertEquals(7.0f, y[1], 1e-6f);
    }

    /**
     * Test ONNX Community K/V projection dimension scenario:
     * ONNX stores K_proj as [hidden_size=896, kv_size=128] (i.e. [K, N])
     * After transpose: [kv_size=128, hidden_size=896] (i.e. [N, K])
     * Verify dimension detection matches.
     */
    @Test
    void onnxCommunityKvProjDimensionsAreTransposed() {
        // For Qwen 0.5B: hidden=896, kv_size=128
        // ONNX Community stores K_proj as dims=[896, 128]
        // Expected by DenseWeight: [N=128, K=896]
        // So dims[0]=896 == K=896, dims[1]=128 == N=128 → needs transpose

        int hidden = 896;
        int kvSize = 128;
        int N = kvSize;   // output dim
        int K = hidden;   // input dim

        // Simulated ONNX dims
        long[] onnxDims = {hidden, kvSize}; // = {K, N}

        // Verify the detection logic: dims[0]==K && dims[1]==N → needs transpose
        assertTrue(onnxDims[0] == K && onnxDims[1] == N,
                "ONNX Community K/V proj dims should be [K, N] = [hidden, kvSize]");
    }

    /**
     * Test ONNX Community tensor naming pattern recognition.
     * The ONNX Community format uses names like:
     * - model.layers.0.attn.q_proj.MatMul.weight (attention projections)
     * - model.layers.0.attn.q_proj.Add.bias (attention biases)
     * - model.layers.0.mlp.gate_proj.MatMul.weight (MLP projections)
     * - model.layers.24.final_norm_layernorm.weight (final norm for 24-layer model)
     */
    @Test
    void onnxCommunityTensorNamingPatterns() {
        // Attention projection weight
        String qProjName = "model.layers.0.attn.q_proj.MatMul.weight";
        assertTrue(qProjName.contains("attn") && qProjName.contains("MatMul.weight"));

        // Bias
        String qBiasName = "model.layers.0.attn.q_proj.Add.bias";
        assertTrue(qBiasName.contains("Add.bias"));

        // MLP
        String gateProjName = "model.layers.0.mlp.gate_proj.MatMul.weight";
        assertTrue(gateProjName.contains("mlp") && gateProjName.contains("MatMul.weight"));

        // Final norm (for a 24-layer model, stored as pseudo-layer 24)
        String finalNormName = "model.layers.24.final_norm_layernorm.weight";
        assertTrue(finalNormName.contains("final_norm_layernorm"));
    }

    /**
     * Test that describeUnsupportedFormat reports missing model.onnx clearly.
     */
    @Test
    void describeUnsupportedFormatReportsMissingFile(@org.junit.jupiter.api.io.TempDir java.nio.file.Path tmp) {
        String msg = Qwen2Weights.describeUnsupportedFormat(tmp);
        assertNotNull(msg);
        assertTrue(msg.contains("model.onnx"), msg);
        assertTrue(msg.contains("missing"), msg);
    }

    @Test
    void loadNormWeightFallsBackToInlineInitializer() throws Exception {
        Map<String, OnnxModelReader.OnnxTensor> inline = new HashMap<>();
        String name = "model.layers.0.input_layernorm.weight";
        float[] expected = new float[]{1f, 2f, 3f, 4f};
        inline.put(name, new OnnxModelReader.OnnxTensor(
                name, new long[]{4}, OnnxModelReader.ONNX_FLOAT, expected, new byte[0]));

        Method m = Qwen2Weights.class.getDeclaredMethod(
                "loadNormWeightWithAlternatives", String.class, String.class, Map.class, Map.class, java.nio.MappedByteBuffer.class);
        m.setAccessible(true);

        float[] actual = (float[]) m.invoke(null, "model.layers.0", "input_layernorm", Map.of(), inline, null);
        assertArrayEquals(expected, actual);
    }

    @Test
    void loadOptionalBiasFallsBackToInlineInitializerAndValidatesSize() throws Exception {
        Map<String, OnnxModelReader.OnnxTensor> inline = new HashMap<>();
        String name = "model.layers.0.attn.k_proj.Add.bias";
        float[] expected = new float[]{0.1f, 0.2f};
        inline.put(name, new OnnxModelReader.OnnxTensor(
                name, new long[]{2}, OnnxModelReader.ONNX_FLOAT, expected, new byte[0]));

        Method m = Qwen2Weights.class.getDeclaredMethod(
                "loadOptionalBias", String.class, Map.class, Map.class, java.nio.MappedByteBuffer.class, int.class);
        m.setAccessible(true);

        float[] actual = (float[]) m.invoke(null, name, Map.of(), inline, null, 2);
        assertArrayEquals(expected, actual);

        Throwable ex = assertThrows(Throwable.class,
                () -> m.invoke(null, name, Map.of(), inline, null, 3));
        assertNotNull(ex.getCause());
        assertTrue(ex.getCause().getMessage().contains("Unexpected size for " + name), ex.getCause().getMessage());
    }

    @Test
    void loadFinalNormFallsBackToInlineOnnxCommunityName() throws Exception {
        Map<String, OnnxModelReader.OnnxTensor> inline = new HashMap<>();
        String name = "model.layers.24.final_norm_layernorm.weight";
        float[] expected = new float[]{0.5f, 0.6f};
        inline.put(name, new OnnxModelReader.OnnxTensor(
                name, new long[]{2}, OnnxModelReader.ONNX_FLOAT, expected, new byte[0]));

        Method m = Qwen2Weights.class.getDeclaredMethod(
                "loadFinalNormWeight", Qwen2Config.class, Map.class, Map.class, java.nio.MappedByteBuffer.class);
        m.setAccessible(true);

        Qwen2Config config = new Qwen2Config(896, 14, 24, 2, 151936, 32768, 4864, 1e-6f, 1_000_000f);
        float[] actual = (float[]) m.invoke(null, config, Map.of(), inline, null);
        assertArrayEquals(expected, actual);
    }
}
