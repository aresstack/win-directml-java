package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.model.TensorCatalog;
import com.aresstack.windirectml.inference.model.TensorEntry;
import com.aresstack.windirectml.inference.model.TensorStorageKind;
import com.aresstack.windirectml.windows.OnnxModelReader;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class T5SafeTensorsLayoutCompilerTest {

    @TempDir
    Path tempDir;

    @Test
    void incompleteSafeTensorsLayoutReportsMissingT5Roles() throws Exception {
        T5Config config = tinyConfig(false);
        Map<String, OnnxTensor> tensors = new LinkedHashMap<>();
        tensors.put("shared.weight", tensor("shared.weight", config.vocabSize(), config.modelSize()));
        T5ModelImport imported = imported(tensors);

        T5LayoutManifest manifest = T5SafeTensorsLayoutCompiler.analyze(imported, config);

        assertEquals(T5SafeTensorsLayoutCompiler.LAYOUT_SCHEMA, manifest.schema());
        assertFalse(manifest.complete());
        assertFalse(manifest.runtimeLoadable());
        assertEquals("t5-safetensors-layout-only", manifest.runtimeLoadMode());
        assertTrue(manifest.missingRequired().contains("encoder.final_layer_norm.weight"));
        assertEquals(2, manifest.roles().size());
        assertTrue(manifest.roles().stream().anyMatch(role -> "lm_head".equals(role.role()) && role.tied()));
    }

    @Test
    void completeCodeT5DenseLayoutBecomesRuntimeLoadable() throws Exception {
        T5Config config = tinyConfig(false);
        Map<String, OnnxTensor> tensors = completeDenseT5Tensors(config);
        T5ModelImport imported = imported(tensors);

        T5LayoutManifest manifest = T5SafeTensorsLayoutCompiler.analyze(imported, config);

        assertTrue(manifest.complete(), manifest.missingRequired().toString());
        assertTrue(manifest.runtimeLoadable(), manifest.unsupportedRuntimeDtypes().toString());
        assertEquals("wdmlpack-native-t5-dense-payload", manifest.runtimeLoadMode());
        assertEquals(27, manifest.roleCount());
        assertTrue(manifest.roles().stream().anyMatch(role -> "lm_head".equals(role.role()) && role.tied()));
        assertTrue(manifest.roles().stream().anyMatch(role ->
                "decoder.layers.000.cross_attn.q.weight".equals(role.runtimeName())));
    }

    @Test
    void wrongTensorShapeIsReportedBeforeRuntimeLoading() throws Exception {
        T5Config config = tinyConfig(false);
        Map<String, OnnxTensor> tensors = completeDenseT5Tensors(config);
        tensors.put("decoder.block.0.layer.1.EncDecAttention.q.weight",
                tensor("decoder.block.0.layer.1.EncDecAttention.q.weight", 99, config.modelSize()));
        T5ModelImport imported = imported(tensors);

        T5LayoutManifest manifest = T5SafeTensorsLayoutCompiler.analyze(imported, config);

        assertFalse(manifest.complete());
        assertFalse(manifest.runtimeLoadable());
        assertEquals(1, manifest.shapeErrors().size());
        assertTrue(manifest.shapeErrors().get(0).contains("EncDecAttention.q.weight"));
    }

    @Test
    void nonSafeTensorsSourceIsNotAnalyzedAsT5SafeTensors() throws Exception {
        T5LayoutManifest manifest = T5SafeTensorsLayoutCompiler.analyze(
                new T5ModelImport("onnx", tempDir.resolve("model.onnx"), Map.of(), new TensorCatalog(List.of())),
                tinyConfig(false));

        assertFalse(manifest.safeTensorsSource());
        assertEquals("not-safetensors", manifest.runtimeLoadMode());
    }

    @Test
    void manifestMapContainsStableT5CompilerMetadata() throws Exception {
        T5Config config = tinyConfig(false);
        T5LayoutManifest manifest = T5SafeTensorsLayoutCompiler.analyze(imported(completeDenseT5Tensors(config)), config);

        Map<String, Object> out = manifest.toManifest();

        assertEquals("huggingface-t5-dense", out.get("sourceLayout"));
        assertEquals(T5SafeTensorsLayoutCompiler.COMPILER_VERSION, out.get("compilerVersion"));
        assertEquals(true, out.get("runtimeLoadable"));
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> roles = (List<Map<String, Object>>) out.get("roles");
        assertEquals(27, roles.size());
    }

    private T5ModelImport imported(Map<String, OnnxTensor> tensors) throws Exception {
        Path model = tempDir.resolve("model.safetensors");
        Files.write(model, new byte[]{1, 2, 3});
        List<TensorEntry> entries = tensors.values().stream()
                .map(tensor -> new TensorEntry(tensor.name(), tensor.dataType(), tensor.dims(),
                        TensorStorageKind.INLINE, tensor.rawByteLength()))
                .toList();
        return new T5ModelImport("safetensors", model, tensors, new TensorCatalog(entries));
    }

    private static Map<String, OnnxTensor> completeDenseT5Tensors(T5Config config) {
        Map<String, OnnxTensor> tensors = new LinkedHashMap<>();
        add(tensors, "shared.weight", config.vocabSize(), config.modelSize());
        add(tensors, "encoder.final_layer_norm.weight", config.modelSize());
        add(tensors, "decoder.final_layer_norm.weight", config.modelSize());
        add(tensors, "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                config.relativeAttentionBuckets(), config.attentionHeads());
        add(tensors, "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                config.relativeAttentionBuckets(), config.attentionHeads());

        addEncoderBlock(tensors, 0, config);
        addDecoderBlock(tensors, 0, config);
        return tensors;
    }

    private static void addEncoderBlock(Map<String, OnnxTensor> tensors, int layer, T5Config config) {
        add(tensors, "encoder.block." + layer + ".layer.0.layer_norm.weight", config.modelSize());
        add(tensors, "encoder.block." + layer + ".layer.1.layer_norm.weight", config.modelSize());
        addAttention(tensors, "encoder.block." + layer + ".layer.0.SelfAttention", config);
        add(tensors, "encoder.block." + layer + ".layer.1.DenseReluDense.wi.weight",
                config.feedForwardSize(), config.modelSize());
        add(tensors, "encoder.block." + layer + ".layer.1.DenseReluDense.wo.weight",
                config.modelSize(), config.feedForwardSize());
    }

    private static void addDecoderBlock(Map<String, OnnxTensor> tensors, int layer, T5Config config) {
        add(tensors, "decoder.block." + layer + ".layer.0.layer_norm.weight", config.modelSize());
        add(tensors, "decoder.block." + layer + ".layer.1.layer_norm.weight", config.modelSize());
        add(tensors, "decoder.block." + layer + ".layer.2.layer_norm.weight", config.modelSize());
        addAttention(tensors, "decoder.block." + layer + ".layer.0.SelfAttention", config);
        addAttention(tensors, "decoder.block." + layer + ".layer.1.EncDecAttention", config);
        add(tensors, "decoder.block." + layer + ".layer.2.DenseReluDense.wi.weight",
                config.feedForwardSize(), config.modelSize());
        add(tensors, "decoder.block." + layer + ".layer.2.DenseReluDense.wo.weight",
                config.modelSize(), config.feedForwardSize());
    }

    private static void addAttention(Map<String, OnnxTensor> tensors, String prefix, T5Config config) {
        add(tensors, prefix + ".q.weight", config.attentionInnerSize(), config.modelSize());
        add(tensors, prefix + ".k.weight", config.attentionInnerSize(), config.modelSize());
        add(tensors, prefix + ".v.weight", config.attentionInnerSize(), config.modelSize());
        add(tensors, prefix + ".o.weight", config.modelSize(), config.attentionInnerSize());
    }

    private static void add(Map<String, OnnxTensor> tensors, String name, long... dims) {
        tensors.put(name, tensor(name, dims));
    }

    private static OnnxTensor tensor(String name, long... dims) {
        int length = Math.toIntExact(elements(dims) * Short.BYTES);
        ByteBuffer payload = ByteBuffer.allocate(length).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < length; i++) {
            payload.put((byte) (i & 0x7f));
        }
        payload.flip();
        return new OnnxTensor(name, dims, OnnxModelReader.ONNX_FLOAT16,
                new float[0], new byte[0], payload.asReadOnlyBuffer(), length);
    }

    private static long elements(long[] dims) {
        long total = 1;
        for (long dim : dims) {
            total *= dim;
        }
        return total;
    }

    private static T5Config tinyConfig(boolean gated) {
        return new T5Config(List.of("T5ForConditionalGeneration"), "t5", true,
                4, 2, 8, 1, 1, 2, 6,
                4, 16, 1e-6f, 0, 2, 0,
                true, gated ? "gated-gelu" : "relu");
    }
}
