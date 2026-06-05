package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.model.ModelSource;
import com.aresstack.windirectml.inference.model.RuntimeModelPackage;
import com.aresstack.windirectml.inference.model.RuntimeTensor;
import com.aresstack.windirectml.inference.model.RuntimeTensorCatalog;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxGraph;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxNode;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Runtime-facing Qwen source backed by a {@code .wdmlpack} package.
 *
 * <p>Model-family specific validation stays here. Generic package reading,
 * manifest access, payload mapping, source fingerprint validation, and tensor
 * catalog reconstruction are handled by {@link RuntimeModelPackage} so T5 and
 * later model families can reuse the same runtime package seam.</p>
 */
final class QwenWdmlPackModelSource implements ModelSource<QwenModelImport> {

    private static final Logger log = LoggerFactory.getLogger(QwenWdmlPackModelSource.class);

    private final Path modelDir;
    private final Path packagePath;
    private final String requestedModelFileName;
    private final Qwen2Config config;

    QwenWdmlPackModelSource(Path modelDir, Path packagePath, String requestedModelFileName, Qwen2Config config) {
        this.modelDir = Objects.requireNonNull(modelDir, "modelDir");
        this.packagePath = Objects.requireNonNull(packagePath, "packagePath").toAbsolutePath().normalize();
        this.requestedModelFileName = QwenModelDirValidator.normalizeModelFileName(requestedModelFileName);
        this.config = Objects.requireNonNull(config, "config");
    }

    @Override
    public String format() {
        return "wdmlpack";
    }

    @Override
    public Path location() {
        return packagePath;
    }

    @Override
    public QwenModelImport load() throws IOException {
        RuntimeModelPackage modelPackage = RuntimeModelPackage.open(packagePath);
        validateModel(modelPackage);
        validateCacheContract(modelPackage);
        modelPackage.validateRuntimeLoadable();
        validateSourceFingerprintIfPresent(modelPackage);

        boolean payloadIncluded = modelPackage.payloadIncluded();
        log.info("Loading Qwen runtime package: {} (mode={}, payloadIncluded={})",
                packagePath.getFileName(), modelPackage.manifest().get("mode"), payloadIncluded);
        if (payloadIncluded) {
            return loadNativePayload(modelPackage);
        }
        return loadManifestOnlyDelegate(modelPackage);
    }

    private QwenModelImport loadManifestOnlyDelegate(RuntimeModelPackage modelPackage) throws IOException {
        Path sourceOnnx = modelPackage.resolveSourcePath(modelDir, requestedModelFileName);
        modelPackage.validateSourceSize(sourceOnnx);
        log.info("wdmlpack front door: using package manifest, tensor payload source={}",
                sourceOnnx.getFileName());

        QwenOnnxModelSource delegate = new QwenOnnxModelSource(sourceOnnx.getParent(), sourceOnnx.getFileName().toString());
        QwenModelImport imported = delegate.load();
        return new QwenModelImport("wdmlpack-manifest", imported.modelPath(), imported.graph(),
                imported.externalRefs(), imported.inlineTensors(), imported.sourceTensorCatalog());
    }

    private QwenModelImport loadNativePayload(RuntimeModelPackage modelPackage) throws IOException {
        RuntimeTensorCatalog runtimeTensors = modelPackage.runtimeTensorCatalog();
        Map<String, OnnxTensor> inlineTensors = adaptRuntimeTensors(runtimeTensors);
        OnnxGraph graph = loadRuntimeGraph(modelPackage.manifest(), inlineTensors);
        log.info("wdmlpack native payload: mapped {} tensors from package payload ({})",
                inlineTensors.size(), QwenWdmlPackCompiler.formatBytes(modelPackage.header().payloadLength()));
        return new QwenModelImport("wdmlpack-payload", packagePath, graph, Map.of(), inlineTensors,
                runtimeTensors.toSourceTensorCatalog());
    }

    @SuppressWarnings("unchecked")
    private void validateModel(RuntimeModelPackage modelPackage) throws IOException {
        Map<String, Object> model = modelPackage.requireMap("model");
        String architecture = String.valueOf(model.get("architecture")).toLowerCase(Locale.ROOT);
        if (!"qwen2".equals(architecture)) {
            throw new IOException("Unsupported wdmlpack architecture: " + model.get("architecture"));
        }
        assertModelInt(model, "hiddenSize", config.hiddenSize());
        assertModelInt(model, "numHiddenLayers", config.numHiddenLayers());
        assertModelInt(model, "numAttentionHeads", config.numAttentionHeads());
        assertModelInt(model, "numKeyValueHeads", config.numKeyValueHeads());
        assertModelInt(model, "headDim", config.headDim());
        assertModelInt(model, "vocabSize", config.vocabSize());
        assertModelInt(model, "intermediateSize", config.intermediateSize());
    }

    private void validateCacheContract(RuntimeModelPackage modelPackage) throws IOException {
        Object cacheObj = modelPackage.manifest().get("cache");
        if (!(cacheObj instanceof Map<?, ?>)) {
            throw new IOException("Stale wdmlpack cache: missing v28 cache contract");
        }
        Map<String, Object> cache = modelPackage.requireMap("cache");
        String schema = RuntimeModelPackage.stringValue(cache.get("schema"));
        if (!QwenWdmlPackCompiler.CACHE_SCHEMA.equals(schema)) {
            throw new IOException("Stale wdmlpack cache schema: " + schema
                    + " (expected " + QwenWdmlPackCompiler.CACHE_SCHEMA + ")");
        }
        int compilerVersion = RuntimeModelPackage.intValue(cache.get("compilerVersion"), -1);
        if (compilerVersion != QwenWdmlPackCompiler.COMPILER_VERSION) {
            throw new IOException("Stale wdmlpack compiler version: " + compilerVersion
                    + " (expected " + QwenWdmlPackCompiler.COMPILER_VERSION + ")");
        }
    }

    private void validateSourceFingerprintIfPresent(RuntimeModelPackage modelPackage) throws IOException {
        Path sourceOnnx = modelPackage.resolveSourcePath(modelDir, requestedModelFileName);
        if (!Files.exists(sourceOnnx)) {
            if (modelPackage.payloadIncluded()) {
                return;
            }
            throw new IOException("wdmlpack source ONNX is missing: " + sourceOnnx);
        }
        modelPackage.validateSourceFingerprint(sourceOnnx);
    }

    private Map<String, OnnxTensor> adaptRuntimeTensors(RuntimeTensorCatalog runtimeTensors) {
        Map<String, OnnxTensor> tensors = new LinkedHashMap<>();
        for (RuntimeTensor tensor : runtimeTensors.values()) {
            if (tensor.hasPayload()) {
                tensors.put(tensor.name(), new OnnxTensor(tensor.name(), tensor.dims(), tensor.dataType(),
                        new float[0], new byte[0], tensor.rawDataBuffer().order(ByteOrder.LITTLE_ENDIAN),
                        tensor.rawByteLength()));
            } else {
                tensors.put(tensor.name(), new OnnxTensor(tensor.name(), tensor.dims(), tensor.dataType(),
                        new float[0], new byte[0]));
            }
        }
        return tensors;
    }

    @SuppressWarnings("unchecked")
    private OnnxGraph loadRuntimeGraph(Map<String, Object> manifest, Map<String, OnnxTensor> initializers) {
        List<OnnxNode> nodes = new ArrayList<>();
        Object runtimeGraphObj = manifest.get("runtimeGraph");
        if (runtimeGraphObj instanceof Map<?, ?> graphRaw) {
            Object nodesObj = RuntimeModelPackage.castMap(graphRaw).get("nodes");
            if (nodesObj instanceof List<?> nodeList) {
                for (Object item : nodeList) {
                    if (!(item instanceof Map<?, ?> nodeRaw)) {
                        continue;
                    }
                    Map<String, Object> node = RuntimeModelPackage.castMap(nodeRaw);
                    String opType = RuntimeModelPackage.stringValue(node.get("opType"));
                    List<String> inputs = RuntimeModelPackage.stringList(node.get("inputs"));
                    List<String> outputs = RuntimeModelPackage.stringList(node.get("outputs"));
                    if (!opType.isBlank()) {
                        nodes.add(new OnnxNode(opType, inputs, outputs, Map.of()));
                    }
                }
            }
        }
        String graphName = "wdmlpack_graph";
        Object sourceObj = manifest.get("source");
        if (sourceObj instanceof Map<?, ?> sourceRaw) {
            graphName = RuntimeModelPackage.stringValue(RuntimeModelPackage.castMap(sourceRaw).get("graphName"));
            if (graphName.isBlank()) graphName = "wdmlpack_graph";
        }
        return new OnnxGraph(graphName, nodes, initializers, List.of(), List.of());
    }

    private static void assertModelInt(Map<String, Object> model, String key, int expected) throws IOException {
        int actual = RuntimeModelPackage.intValue(model.get(key), Integer.MIN_VALUE);
        if (actual != expected) {
            throw new IOException("wdmlpack model metadata mismatch for " + key
                    + ": manifest=" + actual + ", config=" + expected);
        }
    }
}
