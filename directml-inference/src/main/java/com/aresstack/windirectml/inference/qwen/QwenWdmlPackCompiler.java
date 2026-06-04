package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.model.TensorEntry;
import com.aresstack.windirectml.inference.model.WdmlPackWriter;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
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
 * <p>v21 writes only the package manifest and tensor directory metadata. The
 * runtime continues to use the ONNX-backed import path until the payload format
 * is wired in a later release.</p>
 */
final class QwenWdmlPackCompiler {

    private static final Logger log = LoggerFactory.getLogger(QwenWdmlPackCompiler.class);

    static final String PROP_WRITE_MANIFEST = "windirectml.wdmlpack.writeManifest";
    static final String PROP_AUTO_CREATE = "windirectml.wdmlpack.autoCreate";
    static final String PROP_LOAD = "windirectml.wdmlpack.load";
    static final String PROP_OUTPUT = "windirectml.wdmlpack.output";

    private QwenWdmlPackCompiler() {
    }

    static void writeManifestIfRequested(QwenModelImport imported,
                                         Qwen2Config config,
                                         Path modelDir,
                                         String modelFileName) {
        if (Boolean.getBoolean(PROP_WRITE_MANIFEST)) {
            writeManifest(imported, config, modelDir, modelFileName, true);
        }
    }

    static void writeManifestIfAutoCreateEnabled(QwenModelImport imported,
                                                 Qwen2Config config,
                                                 Path modelDir,
                                                 String modelFileName) {
        if (Boolean.parseBoolean(System.getProperty(PROP_AUTO_CREATE, "true"))) {
            writeManifest(imported, config, modelDir, modelFileName, false);
        }
    }

    private static void writeManifest(QwenModelImport imported,
                                      Qwen2Config config,
                                      Path modelDir,
                                      String modelFileName,
                                      boolean explicitRequest) {
        try {
            Path output = resolveOutputPath(modelDir, modelFileName);
            Map<String, Object> manifest = buildManifest(imported, config, modelDir, modelFileName);
            WdmlPackWriter.writeManifestOnly(output, manifest);
            log.info("Wrote manifest-only Qwen wdmlpack{}: {}", explicitRequest ? "" : " cache", output);
        } catch (Exception e) {
            // Package creation is intentionally opportunistic. It must never break
            // the already validated ONNX/WARP inference path.
            log.warn("Could not write optional Qwen wdmlpack manifest; continuing with ONNX runtime path: {}",
                    e.toString());
            log.debug("wdmlpack manifest write failure", e);
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
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("format", "wdmlpack");
        root.put("version", WdmlPackWriter.VERSION);
        root.put("createdAt", Instant.now().toString());
        root.put("mode", "manifest-only");
        root.put("payloadIncluded", false);
        root.put("runtimeLoadable", true);
        root.put("runtimeLoadMode", "wdmlpack-frontdoor-onnx-payload");
        root.put("note", "v22 manifest package: Qwen runtime can start from this package, while tensor payload still delegates to the source ONNX until native payloads are added.");

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
        root.put("tensors", buildTensorDirectory(imported));
        return root;
    }

    private static List<Map<String, Object>> buildTensorDirectory(QwenModelImport imported) {
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
            tensor.put("payloadOffset", -1L); // v22 manifest-only; native tensor payload starts in a later package version

            Qwen2Weights.ExternalTensorRef external = imported.externalRefs().get(entry.name());
            if (external != null) {
                tensor.put("sourceOffset", external.offset());
                tensor.put("sourceLength", external.length());
            }
            tensors.add(tensor);
        }
        return tensors;
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
}
