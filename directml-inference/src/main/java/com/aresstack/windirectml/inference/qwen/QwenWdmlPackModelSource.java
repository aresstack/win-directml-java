package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.model.ModelSource;
import com.aresstack.windirectml.inference.model.TensorCatalog;
import com.aresstack.windirectml.inference.model.TensorEntry;
import com.aresstack.windirectml.inference.model.TensorStorageKind;
import com.aresstack.windirectml.inference.model.WdmlPackWriter;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxGraph;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxNode;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Runtime-facing Qwen source backed by a {@code .wdmlpack} package.
 *
 * <p>v22 manifest-only packages remain a compatibility front door and delegate
 * payloads to ONNX. v23 payload packages reconstruct the tensor catalog and the
 * minimal Qwen runtime graph directly from the package, so ONNX parsing is no
 * longer needed on the hot startup path.</p>
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
        WdmlPackWriter.Header header = WdmlPackWriter.readHeader(packagePath);
        Map<String, Object> manifest = WdmlPackWriter.readManifest(packagePath);
        validateRoot(manifest, header);
        validateModel(manifest);

        boolean payloadIncluded = Boolean.TRUE.equals(manifest.get("payloadIncluded"));
        log.info("Loading Qwen runtime package: {} (mode={}, payloadIncluded={})",
                packagePath.getFileName(), manifest.get("mode"), payloadIncluded);
        if (payloadIncluded) {
            return loadNativePayload(manifest, header);
        }
        return loadManifestOnlyDelegate(manifest);
    }

    private QwenModelImport loadManifestOnlyDelegate(Map<String, Object> manifest) throws IOException {
        Path sourceOnnx = resolveSourceOnnx(manifest);
        validateSource(manifest, sourceOnnx);
        log.info("wdmlpack v22 front door: using package manifest, tensor payload source={}",
                sourceOnnx.getFileName());

        QwenOnnxModelSource delegate = new QwenOnnxModelSource(sourceOnnx.getParent(), sourceOnnx.getFileName().toString());
        QwenModelImport imported = delegate.load();
        return new QwenModelImport("wdmlpack-manifest", imported.modelPath(), imported.graph(),
                imported.externalRefs(), imported.inlineTensors(), imported.tensorCatalog());
    }

    private QwenModelImport loadNativePayload(Map<String, Object> manifest,
                                              WdmlPackWriter.Header header) throws IOException {
        long fileSize = Files.size(packagePath);
        if (fileSize > Integer.MAX_VALUE) {
            throw new IOException("wdmlpack package is too large for the current mmap reader: " + packagePath
                    + " (" + fileSize + " bytes)");
        }
        try (FileChannel channel = FileChannel.open(packagePath, StandardOpenOption.READ)) {
            MappedByteBuffer mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);
            mapped.order(ByteOrder.LITTLE_ENDIAN);
            Map<String, OnnxTensor> inlineTensors = loadPayloadTensors(manifest, header, mapped, fileSize);
            OnnxGraph graph = loadRuntimeGraph(manifest, inlineTensors);
            TensorCatalog catalog = buildCatalog(manifest);
            log.info("wdmlpack v23 native payload: mapped {} tensors from package payload ({})",
                    inlineTensors.size(), QwenWdmlPackCompiler.formatBytes(header.payloadLength()));
            return new QwenModelImport("wdmlpack-payload", packagePath, graph, Map.of(), inlineTensors, catalog);
        }
    }

    private void validateRoot(Map<String, Object> manifest, WdmlPackWriter.Header header) throws IOException {
        if (!"wdmlpack".equals(manifest.get("format"))) {
            throw new IOException("Invalid wdmlpack manifest format: " + manifest.get("format"));
        }
        int version = intValue(manifest.get("version"), -1);
        if (version != WdmlPackWriter.VERSION) {
            throw new IOException("Unsupported wdmlpack manifest version: " + version);
        }
        boolean payloadIncluded = Boolean.TRUE.equals(manifest.get("payloadIncluded"));
        if (payloadIncluded && !header.payloadIncluded()) {
            throw new IOException("wdmlpack manifest says payloadIncluded=true but the container header has no payload");
        }
        if (!payloadIncluded && header.payloadIncluded()) {
            throw new IOException("wdmlpack container has payload but manifest says payloadIncluded=false");
        }
    }

    @SuppressWarnings("unchecked")
    private void validateModel(Map<String, Object> manifest) throws IOException {
        Object modelObj = manifest.get("model");
        if (!(modelObj instanceof Map<?, ?> modelRaw)) {
            throw new IOException("Invalid wdmlpack: missing model metadata");
        }
        Map<String, Object> model = (Map<String, Object>) modelRaw;
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

    @SuppressWarnings("unchecked")
    private Path resolveSourceOnnx(Map<String, Object> manifest) throws IOException {
        Object sourceObj = manifest.get("source");
        if (!(sourceObj instanceof Map<?, ?> sourceRaw)) {
            throw new IOException("Invalid wdmlpack: missing source metadata");
        }
        Map<String, Object> source = (Map<String, Object>) sourceRaw;
        String relativePath = stringValue(source.get("relativePath"));
        String fileName = stringValue(source.get("fileName"));
        String candidate = !relativePath.isBlank() ? relativePath : (!fileName.isBlank() ? fileName : requestedModelFileName);
        Path resolved = modelDir.resolve(candidate).toAbsolutePath().normalize();
        if (!resolved.startsWith(modelDir.toAbsolutePath().normalize())) {
            throw new IOException("Invalid wdmlpack source path escapes model directory: " + candidate);
        }
        return resolved;
    }

    @SuppressWarnings("unchecked")
    private void validateSource(Map<String, Object> manifest, Path sourceOnnx) throws IOException {
        if (!Files.isRegularFile(sourceOnnx)) {
            throw new IOException("wdmlpack source ONNX is missing: " + sourceOnnx);
        }
        Object sourceObj = manifest.get("source");
        if (sourceObj instanceof Map<?, ?> sourceRaw) {
            Map<String, Object> source = (Map<String, Object>) sourceRaw;
            long expectedSize = longValue(source.get("sizeBytes"), -1L);
            long actualSize = Files.size(sourceOnnx);
            if (expectedSize >= 0 && expectedSize != actualSize) {
                throw new IOException("wdmlpack source ONNX size mismatch for " + sourceOnnx.getFileName()
                        + ": manifest=" + expectedSize + ", actual=" + actualSize);
            }
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, OnnxTensor> loadPayloadTensors(Map<String, Object> manifest,
                                                       WdmlPackWriter.Header header,
                                                       MappedByteBuffer mapped,
                                                       long fileSize) throws IOException {
        Object tensorsObj = manifest.get("tensors");
        if (!(tensorsObj instanceof List<?> tensorsRaw)) {
            throw new IOException("Invalid wdmlpack payload: missing tensor directory");
        }
        Map<String, OnnxTensor> tensors = new LinkedHashMap<>();
        for (Object item : tensorsRaw) {
            if (!(item instanceof Map<?, ?> tensorRaw)) {
                continue;
            }
            Map<String, Object> tensor = (Map<String, Object>) tensorRaw;
            String name = stringValue(tensor.get("name"));
            int dataType = intValue(tensor.get("dataType"), 0);
            long[] dims = dimsValue(tensor.get("dims"));
            long byteLength = longValue(tensor.get("byteLength"), 0L);
            long payloadOffset = longValue(tensor.get("payloadOffset"), -1L);
            long payloadLength = longValue(tensor.get("payloadLength"), byteLength);
            if (name.isBlank()) {
                continue;
            }
            if (payloadOffset >= 0 && payloadLength > 0) {
                long absoluteStart = header.payloadOffset() + payloadOffset;
                long absoluteEnd = absoluteStart + payloadLength;
                if (absoluteStart < header.payloadOffset() || absoluteEnd < absoluteStart
                        || absoluteEnd > header.payloadOffset() + header.payloadLength()
                        || absoluteEnd > fileSize || payloadLength > Integer.MAX_VALUE) {
                    throw new IOException("Invalid wdmlpack tensor payload range for " + name
                            + ": offset=" + payloadOffset + ", length=" + payloadLength);
                }
                ByteBuffer slice = mapped.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
                slice.position(Math.toIntExact(absoluteStart));
                slice.limit(Math.toIntExact(absoluteEnd));
                ByteBuffer raw = slice.slice().order(ByteOrder.LITTLE_ENDIAN);
                tensors.put(name, new OnnxTensor(name, dims, dataType, new float[0], new byte[0], raw, (int) payloadLength));
            } else {
                tensors.put(name, new OnnxTensor(name, dims, dataType, new float[0], new byte[0]));
            }
        }
        return tensors;
    }

    @SuppressWarnings("unchecked")
    private OnnxGraph loadRuntimeGraph(Map<String, Object> manifest, Map<String, OnnxTensor> initializers) {
        List<OnnxNode> nodes = new ArrayList<>();
        Object runtimeGraphObj = manifest.get("runtimeGraph");
        if (runtimeGraphObj instanceof Map<?, ?> graphRaw) {
            Object nodesObj = ((Map<String, Object>) graphRaw).get("nodes");
            if (nodesObj instanceof List<?> nodeList) {
                for (Object item : nodeList) {
                    if (!(item instanceof Map<?, ?> nodeRaw)) {
                        continue;
                    }
                    Map<String, Object> node = (Map<String, Object>) nodeRaw;
                    String opType = stringValue(node.get("opType"));
                    List<String> inputs = stringList(node.get("inputs"));
                    List<String> outputs = stringList(node.get("outputs"));
                    if (!opType.isBlank()) {
                        nodes.add(new OnnxNode(opType, inputs, outputs, Map.of()));
                    }
                }
            }
        }
        String graphName = "wdmlpack_graph";
        Object sourceObj = manifest.get("source");
        if (sourceObj instanceof Map<?, ?> sourceRaw) {
            graphName = stringValue(((Map<String, Object>) sourceRaw).get("graphName"));
            if (graphName.isBlank()) graphName = "wdmlpack_graph";
        }
        return new OnnxGraph(graphName, nodes, initializers, List.of(), List.of());
    }

    @SuppressWarnings("unchecked")
    private TensorCatalog buildCatalog(Map<String, Object> manifest) throws IOException {
        Object tensorsObj = manifest.get("tensors");
        if (!(tensorsObj instanceof List<?> tensorsRaw)) {
            throw new IOException("Invalid wdmlpack payload: missing tensor directory");
        }
        List<TensorEntry> entries = new ArrayList<>();
        for (Object item : tensorsRaw) {
            if (!(item instanceof Map<?, ?> tensorRaw)) {
                continue;
            }
            Map<String, Object> tensor = (Map<String, Object>) tensorRaw;
            String name = stringValue(tensor.get("name"));
            if (name.isBlank()) {
                continue;
            }
            int dataType = intValue(tensor.get("dataType"), 0);
            long[] dims = dimsValue(tensor.get("dims"));
            long byteLength = longValue(tensor.get("byteLength"), 0L);
            long payloadOffset = longValue(tensor.get("payloadOffset"), -1L);
            TensorStorageKind kind = payloadOffset >= 0 && byteLength > 0
                    ? TensorStorageKind.INLINE
                    : TensorStorageKind.METADATA_ONLY;
            entries.add(new TensorEntry(name, dataType, dims, kind, byteLength));
        }
        return new TensorCatalog(entries);
    }

    private static void assertModelInt(Map<String, Object> model, String key, int expected) throws IOException {
        int actual = intValue(model.get(key), Integer.MIN_VALUE);
        if (actual != expected) {
            throw new IOException("wdmlpack model metadata mismatch for " + key
                    + ": manifest=" + actual + ", config=" + expected);
        }
    }

    private static int intValue(Object value, int defaultValue) {
        if (value instanceof Number n) return n.intValue();
        if (value instanceof String s && !s.isBlank()) {
            try {
                return Integer.parseInt(s);
            } catch (NumberFormatException ignored) {
                return defaultValue;
            }
        }
        return defaultValue;
    }

    private static long longValue(Object value, long defaultValue) {
        if (value instanceof Number n) return n.longValue();
        if (value instanceof String s && !s.isBlank()) {
            try {
                return Long.parseLong(s);
            } catch (NumberFormatException ignored) {
                return defaultValue;
            }
        }
        return defaultValue;
    }

    private static String stringValue(Object value) {
        return value instanceof String s ? s : "";
    }

    private static List<String> stringList(Object value) {
        if (!(value instanceof List<?> list)) {
            return List.of();
        }
        List<String> out = new ArrayList<>(list.size());
        for (Object item : list) {
            out.add(String.valueOf(item));
        }
        return out;
    }

    private static long[] dimsValue(Object value) {
        if (!(value instanceof List<?> list)) {
            return new long[0];
        }
        long[] dims = new long[list.size()];
        for (int i = 0; i < list.size(); i++) {
            Object item = list.get(i);
            if (item instanceof Number n) dims[i] = n.longValue();
            else if (item instanceof String s && !s.isBlank()) {
                try {
                    dims[i] = Long.parseLong(s);
                } catch (NumberFormatException ignored) {
                    dims[i] = 0L;
                }
            }
        }
        return dims;
    }
}
