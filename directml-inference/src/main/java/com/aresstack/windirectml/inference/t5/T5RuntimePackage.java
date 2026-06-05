package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.model.WdmlPackWriter;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * T5-local package metadata reader for manifest-only wdmlpack artifacts.
 */
public final class T5RuntimePackage {
    private final Path packagePath;
    private final Map<String, Object> manifest;
    private final T5PackageMetadata metadata;

    private T5RuntimePackage(Path packagePath, Map<String, Object> manifest, T5PackageMetadata metadata) {
        this.packagePath = packagePath;
        this.manifest = Map.copyOf(new LinkedHashMap<>(manifest));
        this.metadata = Objects.requireNonNull(metadata, "metadata");
    }

    public static T5RuntimePackage open(Path packagePath) throws IOException {
        Path normalized = Objects.requireNonNull(packagePath, "packagePath").toAbsolutePath().normalize();
        Map<String, Object> manifest = WdmlPackWriter.readManifest(normalized);
        validateT5Manifest(manifest);
        return new T5RuntimePackage(normalized, manifest, metadataFromManifest(manifest));
    }

    public static T5RuntimePackage fromMetadata(T5PackageMetadata metadata) {
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("format", "wdmlpack");
        manifest.put("version", WdmlPackWriter.VERSION);
        manifest.put("modelFamily", T5PackageMetadata.MODEL_FAMILY);
        manifest.put("architecture", T5PackageMetadata.ARCHITECTURE);
        manifest.put("payloadIncluded", false);
        manifest.put("runtimeLoadable", false);
        manifest.put("runtimeLoadMode", T5ManifestPayloadPolicy.RUNTIME_LOAD_MODE);
        manifest.put("reason", T5ManifestPayloadPolicy.REASON);
        manifest.put("t5", metadata.toManifest());
        return new T5RuntimePackage(null, manifest, metadata);
    }

    public Path packagePath() {
        return packagePath;
    }

    public Map<String, Object> manifest() {
        return manifest;
    }

    public T5PackageMetadata metadata() {
        return metadata;
    }

    public boolean runtimeLoadable() {
        return Boolean.TRUE.equals(manifest.get("runtimeLoadable"));
    }

    static void validateT5Manifest(Map<String, Object> manifest) throws IOException {
        String modelFamily = stringValue(manifest.get("modelFamily"));
        if (modelFamily.isBlank() && manifest.get("t5") instanceof Map<?, ?> t5) {
            modelFamily = stringValue(t5.get("modelFamily"));
        }
        if (!T5PackageMetadata.MODEL_FAMILY.equals(modelFamily)) {
            throw new IOException("Not a T5 wdmlpack manifest: modelFamily=" + modelFamily);
        }
        String architecture = stringValue(manifest.get("architecture"));
        if (!architecture.isBlank() && !T5PackageMetadata.ARCHITECTURE.equals(architecture)) {
            throw new IOException("Not a T5 encoder-decoder wdmlpack manifest: architecture=" + architecture);
        }
    }

    @SuppressWarnings("unchecked")
    private static T5PackageMetadata metadataFromManifest(Map<String, Object> manifest) throws IOException {
        Object t5 = manifest.get("t5");
        if (!(t5 instanceof Map<?, ?> raw)) {
            throw new IOException("Invalid T5 wdmlpack manifest: missing t5 section");
        }
        Map<String, Object> map = (Map<String, Object>) raw;
        T5Config config = new T5Config(java.util.List.of("T5ForConditionalGeneration"), "t5", true,
                intValue(map.get("dModel")), intValue(map.get("dKv")), intValue(map.get("dFf")),
                intValue(map.get("encoderLayers")), intValue(map.get("decoderLayers")), intValue(map.get("numHeads")),
                intValue(map.get("vocabSize")), intValue(map.get("relativeAttentionBuckets")),
                intValue(map.get("relativeAttentionMaxDistance")), 0.0f,
                intValue(map.get("decoderStartTokenId")), intValue(map.get("eosTokenId")), intValue(map.get("padTokenId")),
                Boolean.valueOf(String.valueOf(map.getOrDefault("tieWordEmbeddings", Boolean.TRUE))),
                stringValue(map.get("feedForwardProjection")));
        try {
            config.validate();
        } catch (IOException e) {
            throw new IOException("Invalid T5 package metadata: " + e.getMessage(), e);
        }
        return T5PackageMetadata.from(config);
    }

    private static int intValue(Object value) {
        if (value instanceof Number n) {
            return n.intValue();
        }
        if (value instanceof String s && !s.isBlank()) {
            return Integer.parseInt(s);
        }
        return 0;
    }

    private static String stringValue(Object value) {
        return value instanceof String s ? s : "";
    }
}
