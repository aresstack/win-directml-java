package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.model.SourceFingerprint;
import com.aresstack.windirectml.inference.model.WdmlPackWriter;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Writes the manifest-only T5 wdmlpack package used by the parallel T5 strand.
 */
final class T5WdmlPackManifestWriter {

    Path writeManifestOnly(Path output,
                           Path modelDir,
                           T5ModelImport imported,
                           T5Config config,
                           T5LayoutManifest layout) throws IOException {
        return WdmlPackWriter.writeManifestOnly(output, buildManifest(modelDir, imported, config, layout));
    }

    Map<String, Object> buildManifest(Path modelDir,
                                      T5ModelImport imported,
                                      T5Config config,
                                      T5LayoutManifest layout) throws IOException {
        Objects.requireNonNull(modelDir, "modelDir");
        Objects.requireNonNull(imported, "imported");
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(layout, "layout");

        Map<String, Object> root = new LinkedHashMap<>();
        root.put("format", "wdmlpack");
        root.put("version", WdmlPackWriter.VERSION);
        root.put("createdAt", Instant.now().toString());
        root.put("modelFamily", T5PackageMetadata.MODEL_FAMILY);
        root.put("architecture", T5PackageMetadata.ARCHITECTURE);
        root.put("sourceFormat", imported.sourceFormat());
        root.put("sourceLayout", layout.sourceLayout());
        root.put("payloadIncluded", T5ManifestPayloadPolicy.PAYLOAD_INCLUDED);
        root.put("runtimeLoadable", T5ManifestPayloadPolicy.RUNTIME_LOADABLE);
        root.put("runtimeLoadMode", T5ManifestPayloadPolicy.RUNTIME_LOAD_MODE);
        root.put("reason", T5ManifestPayloadPolicy.REASON);
        root.put("source", buildSource(modelDir, imported));
        root.put("model", buildModelSection(config));
        root.put("t5", T5PackageMetadata.from(config).toManifest());
        root.put("layout", layout.toManifest());
        root.put("tensors", layout.roles().stream().map(T5TensorRole::toManifest).toList());
        return root;
    }

    private Map<String, Object> buildSource(Path modelDir, T5ModelImport imported) throws IOException {
        Map<String, Object> source = new LinkedHashMap<>();
        Path root = modelDir.toAbsolutePath().normalize();
        Path sourcePath = imported.modelPath().toAbsolutePath().normalize();
        source.put("format", imported.sourceFormat());
        source.put("fileName", sourcePath.getFileName().toString());
        source.put("relativePath", sourcePath.startsWith(root) ? root.relativize(sourcePath).toString().replace('\\', '/') : sourcePath.getFileName().toString());
        if (java.nio.file.Files.isRegularFile(sourcePath)) {
            SourceFingerprint fingerprint = SourceFingerprint.read(sourcePath);
            source.put("sizeBytes", fingerprint.sizeBytes());
            source.put("lastModifiedMillis", fingerprint.lastModifiedMillis());
            source.put("fileKey", fingerprint.fileKey());
            source.put("fingerprint", fingerprint.value());
        }
        return source;
    }

    private Map<String, Object> buildModelSection(T5Config config) {
        Map<String, Object> model = new LinkedHashMap<>();
        model.put("architecture", T5PackageMetadata.ARCHITECTURE);
        model.put("modelFamily", T5PackageMetadata.MODEL_FAMILY);
        model.put("dModel", config.modelSize());
        model.put("dKv", config.keyValueSize());
        model.put("dFf", config.feedForwardSize());
        model.put("numHeads", config.attentionHeads());
        model.put("encoderLayers", config.encoderLayers());
        model.put("decoderLayers", config.effectiveDecoderLayers());
        model.put("vocabSize", config.vocabSize());
        model.put("feedForwardProjection", config.effectiveFeedForwardProjection());
        model.put("tieWordEmbeddings", config.usesTiedWordEmbeddings());
        return model;
    }
}
