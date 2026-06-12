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
        assertEquals(T5ManifestPayloadPolicy.MODE_WEIGHTS_NOT_LOADABLE, manifest.runtimeLoadMode());
        assertTrue(manifest.missingRequired().contains("encoder.final_layer_norm.weight"));
        assertEquals(1, manifest.roles().size());
    }

    @Test
    void completeCodeT5DenseLayoutIsRuntimeLoadableButNotExecutableYet() throws Exception {
        T5Config config = tinyConfig(false);
        Map<String, OnnxTensor> tensors = completeDenseT5Tensors(config);
        T5ModelImport imported = imported(tensors);

        T5LayoutManifest manifest = T5SafeTensorsLayoutCompiler.analyze(imported, config);

        assertTrue(manifest.complete(), manifest.missingRequired().toString());
        // T5-1: a complete, supported layout is honestly runtime-loadable (not "runtime not implemented").
        assertTrue(manifest.runtimeLoadable());
        assertEquals(T5ManifestPayloadPolicy.MODE_RUNTIME_LOADABLE_NOT_EXECUTABLE, manifest.runtimeLoadMode());
        assertEquals(T5ManifestPayloadPolicy.REASON_RUNTIME_LOADABLE_NOT_EXECUTABLE, manifest.reason());
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


    @Test
    void acceptsGatedFeedForwardLayout() throws Exception {
        T5Config config = tinyConfig(true);
        T5LayoutManifest manifest = T5SafeTensorsLayoutCompiler.analyze(imported(completeDenseT5Tensors(config)), config);

        assertTrue(manifest.complete(), manifest.missingRequired().toString());
        assertTrue(manifest.runtimeLoadable());
        assertTrue(manifest.roles().stream().anyMatch(role -> "encoder.layer.0.ffn.wi_0".equals(role.role())));
        assertTrue(manifest.roles().stream().anyMatch(role -> "decoder.layer.0.ffn.wi_1".equals(role.role())));
    }

    @Test
    void rejectsMissingLmHeadWhenEmbeddingsAreNotTied() throws Exception {
        T5Config config = new T5Config(List.of("T5ForConditionalGeneration"), "t5", true,
                4, 2, 8, 1, 1, 2, 6, 4, 16, 1e-6f,
                0, 2, 0, false, "relu");
        Map<String, OnnxTensor> tensors = completeDenseT5Tensors(config);
        tensors.remove("lm_head.weight");

        T5LayoutManifest manifest = T5SafeTensorsLayoutCompiler.analyze(imported(tensors), config);

        assertFalse(manifest.complete());
        assertTrue(manifest.missingRequired().contains("lm_head.weight"));
    }

    @Test
    void acceptsExplicitLmHeadWhenEmbeddingsAreNotTied() throws Exception {
        T5Config config = new T5Config(List.of("T5ForConditionalGeneration"), "t5", true,
                4, 2, 8, 1, 1, 2, 6, 4, 16, 1e-6f,
                0, 2, 0, false, "relu");

        T5LayoutManifest manifest = T5SafeTensorsLayoutCompiler.analyze(imported(completeDenseT5Tensors(config)), config);

        assertTrue(manifest.complete(), manifest.missingRequired().toString());
        assertTrue(manifest.roles().stream().anyMatch(role -> "lm_head".equals(role.role()) && !role.tied()));
    }

    @Test
    void reportsUnsupportedDtypes() throws Exception {
        T5Config config = tinyConfig(false);
        Map<String, OnnxTensor> tensors = completeDenseT5Tensors(config);
        tensors.put("shared.weight", tensor("shared.weight", OnnxModelReader.ONNX_INT8,
                config.vocabSize(), config.modelSize()));

        T5LayoutManifest manifest = T5SafeTensorsLayoutCompiler.analyze(imported(tensors), config);

        assertTrue(manifest.complete(), manifest.shapeErrors().toString());
        assertFalse(manifest.runtimeLoadable());
        assertEquals(1, manifest.unsupportedRuntimeDtypes().size());
        assertTrue(manifest.unsupportedRuntimeDtypes().get(0).contains("INT8"));
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
        return T5TestFixtures.completeDenseT5Tensors(config);
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


    private static OnnxTensor tensor(String name, int dataType, long... dims) {
        int length = Math.toIntExact(elements(dims) * Short.BYTES);
        ByteBuffer payload = ByteBuffer.allocate(length).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < length; i++) {
            payload.put((byte) (i & 0x7f));
        }
        payload.flip();
        return new OnnxTensor(name, dims, dataType,
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
        return T5TestFixtures.tinyConfig(gated);
    }
}
