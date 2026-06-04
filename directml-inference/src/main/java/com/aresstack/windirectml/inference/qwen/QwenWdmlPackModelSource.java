package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.model.ModelSource;
import com.aresstack.windirectml.inference.model.WdmlPackWriter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Runtime-facing Qwen source backed by a {@code .wdmlpack} package.
 *
 * <p>v22 deliberately treats the package as the runtime front door while the
 * tensor payload still lives in the original ONNX file referenced by the
 * manifest. That gives us the load boundary and package validation without
 * risking the already validated v20 mmap/heap-light tensor path. A later
 * package version can replace the delegate with a native payload reader while
 * keeping this source contract unchanged.</p>
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
        Map<String, Object> manifest = WdmlPackWriter.readManifest(packagePath);
        validateRoot(manifest);
        validateModel(manifest);
        Path sourceOnnx = resolveSourceOnnx(manifest);
        validateSource(manifest, sourceOnnx);

        log.info("Loading Qwen runtime package: {} (mode={}, payloadIncluded={})",
                packagePath.getFileName(), manifest.get("mode"), manifest.get("payloadIncluded"));
        log.info("wdmlpack v22 front door: using package manifest, tensor payload source={}",
                sourceOnnx.getFileName());

        QwenOnnxModelSource delegate = new QwenOnnxModelSource(sourceOnnx.getParent(), sourceOnnx.getFileName().toString());
        QwenModelImport imported = delegate.load();
        return new QwenModelImport(format(), imported.modelPath(), imported.graph(),
                imported.externalRefs(), imported.inlineTensors(), imported.tensorCatalog());
    }

    private void validateRoot(Map<String, Object> manifest) throws IOException {
        if (!"wdmlpack".equals(manifest.get("format"))) {
            throw new IOException("Invalid wdmlpack manifest format: " + manifest.get("format"));
        }
        int version = intValue(manifest.get("version"), -1);
        if (version != WdmlPackWriter.VERSION) {
            throw new IOException("Unsupported wdmlpack manifest version: " + version);
        }
        if (!Boolean.FALSE.equals(manifest.get("payloadIncluded"))) {
            throw new IOException("This runtime only supports v22 manifest-only wdmlpack payloads for now");
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
}
