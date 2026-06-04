package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.model.TensorEntry;
import com.aresstack.windirectml.inference.model.TensorStorageKind;
import com.aresstack.windirectml.inference.model.WdmlPackWriter;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxNode;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Qwen-specific compiler front-end for the internal {@code .wdmlpack} format.
 *
 * <p>v23 can write a payload-carrying package. Existing v22 manifest-only
 * packages remain accepted as a compatibility front door and delegate tensor
 * payloads back to ONNX.</p>
 */
final class QwenWdmlPackCompiler {

    private static final Logger log = LoggerFactory.getLogger(QwenWdmlPackCompiler.class);

    static final String PROP_WRITE_MANIFEST = "windirectml.wdmlpack.writeManifest";
    static final String PROP_AUTO_CREATE = "windirectml.wdmlpack.autoCreate";
    static final String PROP_LOAD = "windirectml.wdmlpack.load";
    static final String PROP_OUTPUT = "windirectml.wdmlpack.output";
    static final String PROP_PAYLOAD = "windirectml.wdmlpack.payload";

    private QwenWdmlPackCompiler() {
    }

    static void writeManifestIfRequested(QwenModelImport imported,
                                         Qwen2Config config,
                                         Path modelDir,
                                         String modelFileName) {
        if (Boolean.getBoolean(PROP_WRITE_MANIFEST)) {
            writePackage(imported, config, modelDir, modelFileName, true);
        }
    }

    static void writeManifestIfAutoCreateEnabled(QwenModelImport imported,
                                                 Qwen2Config config,
                                                 Path modelDir,
                                                 String modelFileName) {
        if (Boolean.parseBoolean(System.getProperty(PROP_AUTO_CREATE, "true"))) {
            writePackage(imported, config, modelDir, modelFileName, false);
        }
    }

    private static void writePackage(QwenModelImport imported,
                                     Qwen2Config config,
                                     Path modelDir,
                                     String modelFileName,
                                     boolean explicitRequest) {
        try {
            Path output = resolveOutputPath(modelDir, modelFileName);
            boolean writePayload = Boolean.parseBoolean(System.getProperty(PROP_PAYLOAD, "true"));
            if (writePayload) {
                PayloadPlan payloadPlan = planPayload(imported);
                Map<String, Object> manifest = buildPayloadManifest(imported, config, modelDir, modelFileName, payloadPlan);
                WdmlPackWriter.writeWithPayload(output, manifest, payloadPlan.entries(), payloadPlan.payloadLength());
                log.info("Wrote payload-included Qwen wdmlpack{}: {} (payload={})",
                        explicitRequest ? "" : " cache", output, formatBytes(payloadPlan.payloadLength()));
            } else {
                Map<String, Object> manifest = buildManifest(imported, config, modelDir, modelFileName);
                WdmlPackWriter.writeManifestOnly(output, manifest);
                log.info("Wrote manifest-only Qwen wdmlpack{}: {}", explicitRequest ? "" : " cache", output);
            }
        } catch (Exception e) {
            // Package creation is intentionally opportunistic. It must never break
            // the already validated ONNX/WARP inference path.
            log.warn("Could not write optional Qwen wdmlpack; continuing with current runtime path: {}", e.toString());
            log.debug("wdmlpack write failure", e);
        }
    }

    static boolean shouldLoadPackage() {
        return Boolean.parseBoolean(System.getProperty(PROP_LOAD, "true"));
    }

    static Path resolveOutputPath(Path modelDir, String modelFileName) {
        String explicit = System.getProperty(PROP_OUTPUT);
        if (explicit != null && !explicit.isBlank()) {
            return Path.of(explicit).toAbsolutePath().normalize();
        }
        String normalized = QwenModelDirValidator.normalizeModelFileName(modelFileName);
        String base = normalized.endsWith(".onnx")
                ? normalized.substring(0, normalized.length() - ".onnx".length())
                : normalized;
        return modelDir.resolve(base + ".wdmlpack").toAbsolutePath().normalize();
    }

    static Map<String, Object> buildManifest(QwenModelImport imported,
                                             Qwen2Config config,
                                             Path modelDir,
                                             String modelFileName) throws IOException {
        Map<String, Object> root = buildBaseManifest(imported, config, modelDir, modelFileName);
        root.put("mode", "manifest-only");
        root.put("payloadIncluded", false);
        root.put("runtimeLoadable", true);
        root.put("runtimeLoadMode", "wdmlpack-frontdoor-onnx-payload");
        root.put("note", "v22/v23 compatibility manifest: Qwen runtime can start from this package, while tensor payload delegates to the source ONNX.");
        root.put("tensors", buildTensorDirectory(imported, null));
        return root;
    }

    static Map<String, Object> buildPayloadManifest(QwenModelImport imported,
                                                    Qwen2Config config,
                                                    Path modelDir,
                                                    String modelFileName,
                                                    PayloadPlan payloadPlan) throws IOException {
        Map<String, Object> root = buildBaseManifest(imported, config, modelDir, modelFileName);
        root.put("mode", "payload");
        root.put("payloadIncluded", true);
        root.put("runtimeLoadable", true);
        root.put("runtimeLoadMode", "wdmlpack-native-payload");
        root.put("payloadAlignment", WdmlPackWriter.PAYLOAD_ALIGNMENT);
        root.put("payloadBytes", payloadPlan.payloadLength());
        root.put("note", "v23 payload package: Qwen runtime can reconstruct the tensor catalog and minimal graph without parsing ONNX.");
        root.put("runtimeGraph", buildRuntimeGraph(imported.graph().nodes()));
        root.put("tensors", buildTensorDirectory(imported, payloadPlan.offsets()));
        return root;
    }

    private static Map<String, Object> buildBaseManifest(QwenModelImport imported,
                                                         Qwen2Config config,
                                                         Path modelDir,
                                                         String modelFileName) throws IOException {
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("format", "wdmlpack");
        root.put("version", WdmlPackWriter.VERSION);
        root.put("createdAt", Instant.now().toString());

        Path source = imported.modelPath();
        Map<String, Object> sourceInfo = new LinkedHashMap<>();
        sourceInfo.put("format", imported.sourceFormat());
        sourceInfo.put("fileName", source.getFileName().toString());
        sourceInfo.put("relativePath", safeRelativize(modelDir, source));
        sourceInfo.put("sizeBytes", Files.exists(source) ? Files.size(source) : -1L);
        sourceInfo.put("graphName", imported.graph().name());
        sourceInfo.put("graphNodes", imported.graph().nodes().size());
        sourceInfo.put("initializers", imported.graph().initializers().size());
        sourceInfo.put("matMulNBitsNodes", imported.graph().nodes().stream()
                .filter(n -> "MatMulNBits".equals(n.opType()))
                .count());
        root.put("source", sourceInfo);

        Map<String, Object> model = new LinkedHashMap<>();
        model.put("architecture", "qwen2");
        model.put("modelFile", QwenModelDirValidator.normalizeModelFileName(modelFileName));
        model.put("hiddenSize", config.hiddenSize());
        model.put("numHiddenLayers", config.numHiddenLayers());
        model.put("numAttentionHeads", config.numAttentionHeads());
        model.put("numKeyValueHeads", config.numKeyValueHeads());
        model.put("headDim", config.headDim());
        model.put("vocabSize", config.vocabSize());
        model.put("intermediateSize", config.intermediateSize());
        model.put("maxPositionEmbeddings", config.maxPositionEmbeddings());
        model.put("rmsNormEps", config.rmsNormEps());
        model.put("ropeTheta", config.ropeTheta());
        model.put("tieWordEmbeddings", config.tieWordEmbeddings());
        root.put("model", model);

        Map<String, Object> catalog = new LinkedHashMap<>();
        catalog.put("count", imported.tensorCatalog().size());
        catalog.put("inlineBytes", imported.tensorCatalog().inlineBytes());
        catalog.put("externalBytes", imported.tensorCatalog().externalBytes());
        catalog.put("metadataOnlyCount", imported.tensorCatalog().metadataOnlyCount());
        root.put("tensorCatalog", catalog);

        root.put("operatorCatalog", buildOperatorCatalog(imported.graph().nodes()));
        return root;
    }

    private static PayloadPlan planPayload(QwenModelImport imported) throws IOException {
        List<TensorEntry> entries = new ArrayList<>(imported.tensorCatalog().entries().values());
        entries.sort(Comparator.comparing(TensorEntry::name));
        Map<String, Long> offsets = new LinkedHashMap<>();
        List<WdmlPackWriter.PayloadEntry> payloadEntries = new ArrayList<>();
        long cursor = 0;
        for (TensorEntry entry : entries) {
            TensorPayload payload = resolvePayload(entry, imported);
            if (payload == null || payload.length() <= 0) {
                offsets.put(entry.name(), -1L);
                continue;
            }
            cursor = WdmlPackWriter.align(cursor, 64);
            long relativeOffset = cursor;
            offsets.put(entry.name(), relativeOffset);
            payloadEntries.add(new WdmlPackWriter.PayloadEntry(entry.name(), relativeOffset, payload.length(), payload.writer()));
            cursor += payload.length();
        }
        return new PayloadPlan(offsets, payloadEntries, cursor);
    }

    private static TensorPayload resolvePayload(TensorEntry entry, QwenModelImport imported) throws IOException {
        OnnxTensor inline = imported.inlineTensors().get(entry.name());
        if (inline != null) {
            if (inline.rawByteLength() > 0) {
                ByteBuffer raw = inline.rawDataBuffer().order(ByteOrder.LITTLE_ENDIAN);
                return new TensorPayload(inline.rawByteLength(), channel -> writeBuffer(channel, raw));
            }
            if (inline.data() != null && inline.data().length > 0) {
                float[] data = inline.data();
                long length = (long) data.length * Float.BYTES;
                return new TensorPayload(length, channel -> writeFloatArray(channel, data));
            }
        }

        Qwen2Weights.ExternalTensorRef external = imported.externalRefs().get(entry.name());
        if (external != null) {
            Path dataPath = QwenModelDirValidator.resolveExternalDataPath(imported.modelPath().getParent());
            return new TensorPayload(external.length(), channel -> copyFileRegion(dataPath, external.offset(), external.length(), channel));
        }
        return null;
    }

    private static void writeBuffer(FileChannel channel, ByteBuffer source) throws IOException {
        ByteBuffer duplicate = source.asReadOnlyBuffer();
        duplicate.position(0);
        while (duplicate.hasRemaining()) {
            channel.write(duplicate);
        }
    }

    private static void writeFloatArray(FileChannel channel, float[] data) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(Math.min(1024 * 1024, Math.max(Float.BYTES, data.length * Float.BYTES)))
                .order(ByteOrder.LITTLE_ENDIAN);
        for (float value : data) {
            if (buffer.remaining() < Float.BYTES) {
                buffer.flip();
                while (buffer.hasRemaining()) channel.write(buffer);
                buffer.clear();
            }
            buffer.putFloat(value);
        }
        buffer.flip();
        while (buffer.hasRemaining()) channel.write(buffer);
    }

    private static void copyFileRegion(Path source, long offset, long length, FileChannel target) throws IOException {
        try (FileChannel input = FileChannel.open(source, StandardOpenOption.READ)) {
            long copied = 0;
            while (copied < length) {
                long n = input.transferTo(offset + copied, length - copied, target);
                if (n <= 0) {
                    ByteBuffer fallback = ByteBuffer.allocate((int) Math.min(1024 * 1024, length - copied));
                    input.read(fallback, offset + copied);
                    fallback.flip();
                    n = target.write(fallback);
                    if (n <= 0) throw new IOException("Could not copy payload region from " + source);
                }
                copied += n;
            }
        }
    }

    private static List<Map<String, Object>> buildTensorDirectory(QwenModelImport imported, Map<String, Long> payloadOffsets) {
        List<TensorEntry> entries = new ArrayList<>(imported.tensorCatalog().entries().values());
        entries.sort(Comparator.comparing(TensorEntry::name));
        List<Map<String, Object>> tensors = new ArrayList<>(entries.size());
        for (TensorEntry entry : entries) {
            Map<String, Object> tensor = new LinkedHashMap<>();
            tensor.put("name", entry.name());
            tensor.put("dataType", entry.dataType());
            tensor.put("dataTypeName", onnxDataTypeName(entry.dataType()));
            tensor.put("dims", toList(entry.dims()));
            tensor.put("storageKind", entry.storageKind().name());
            tensor.put("byteLength", entry.byteLength());
            long payloadOffset = payloadOffsets != null ? payloadOffsets.getOrDefault(entry.name(), -1L) : -1L;
            tensor.put("payloadOffset", payloadOffset);
            tensor.put("payloadLength", payloadOffset >= 0 ? entry.byteLength() : 0L);

            Qwen2Weights.ExternalTensorRef external = imported.externalRefs().get(entry.name());
            if (external != null) {
                tensor.put("sourceOffset", external.offset());
                tensor.put("sourceLength", external.length());
            }
            tensors.add(tensor);
        }
        return tensors;
    }

    private static Map<String, Object> buildRuntimeGraph(List<OnnxNode> nodes) {
        Map<String, Object> graph = new LinkedHashMap<>();
        List<Map<String, Object>> runtimeNodes = new ArrayList<>();
        for (OnnxNode node : nodes) {
            if ("MatMulNBits".equals(node.opType()) || "Add".equals(node.opType())) {
                Map<String, Object> item = new LinkedHashMap<>();
                item.put("opType", node.opType());
                item.put("inputs", node.inputs());
                item.put("outputs", node.outputs());
                runtimeNodes.add(item);
            }
        }
        graph.put("nodes", runtimeNodes);
        graph.put("note", "Minimal Qwen runtime graph; enough for MatMulNBits projection mapping and connected Add biases.");
        return graph;
    }

    private static Map<String, Object> buildOperatorCatalog(List<OnnxNode> nodes) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (OnnxNode node : nodes) {
            counts.merge(node.opType(), 1, Integer::sum);
        }
        Map<String, Object> ops = new LinkedHashMap<>();
        ops.put("counts", counts);
        ops.put("runtimeTarget", "directml-warp-auto");
        ops.put("genericOnnxExecution", false);
        return ops;
    }

    private static List<Long> toList(long[] dims) {
        List<Long> out = new ArrayList<>(dims.length);
        for (long dim : dims) {
            out.add(dim);
        }
        return out;
    }

    private static String onnxDataTypeName(int dataType) {
        return switch (dataType) {
            case 1 -> "FLOAT";
            case 2 -> "UINT8";
            case 3 -> "INT8";
            case 6 -> "INT32";
            case 7 -> "INT64";
            case 10 -> "FLOAT16";
            default -> "ONNX_TYPE_" + dataType;
        };
    }

    private static String safeRelativize(Path root, Path file) {
        try {
            return root.toAbsolutePath().normalize().relativize(file.toAbsolutePath().normalize()).toString()
                    .replace('\\', '/');
        } catch (RuntimeException e) {
            return file.toAbsolutePath().normalize().toString();
        }
    }

    static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        double kb = bytes / 1024.0;
        if (kb < 1024) return String.format(Locale.ROOT, "%.1f KiB", kb);
        double mb = kb / 1024.0;
        if (mb < 1024) return String.format(Locale.ROOT, "%.1f MiB", mb);
        double gb = mb / 1024.0;
        return String.format(Locale.ROOT, "%.2f GiB", gb);
    }

    record PayloadPlan(Map<String, Long> offsets,
                       List<WdmlPackWriter.PayloadEntry> entries,
                       long payloadLength) {
    }

    private record TensorPayload(long length, WdmlPackWriter.PayloadWriter writer) {
    }
}
