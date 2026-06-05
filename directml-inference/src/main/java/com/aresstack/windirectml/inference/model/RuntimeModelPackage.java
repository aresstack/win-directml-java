package com.aresstack.windirectml.inference.model;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Format-neutral descriptor for an internal {@code .wdmlpack} runtime package.
 *
 * <p>The descriptor owns package-level metadata and source-cache validation.
 * File mapping and tensor payload slicing are delegated to {@link WdmlPackReader};
 * parsed JSON access is delegated to {@link WdmlPackManifest}. This keeps model
 * families away from container layout details.</p>
 */
public final class RuntimeModelPackage {

    private final Path packagePath;
    private final WdmlPackWriter.Header header;
    private final WdmlPackManifest manifest;
    private final long fileSize;

    RuntimeModelPackage(Path packagePath,
                        WdmlPackWriter.Header header,
                        WdmlPackManifest manifest,
                        long fileSize) {
        this.packagePath = Objects.requireNonNull(packagePath, "packagePath");
        this.header = Objects.requireNonNull(header, "header");
        this.manifest = Objects.requireNonNull(manifest, "manifest");
        this.fileSize = fileSize;
    }

    public static RuntimeModelPackage open(Path packagePath) throws IOException {
        return WdmlPackReader.open(packagePath);
    }

    public Path packagePath() {
        return packagePath;
    }

    public WdmlPackWriter.Header header() {
        return header;
    }

    public Map<String, Object> manifest() {
        return manifest.root();
    }

    public WdmlPackManifest manifestMetadata() {
        return manifest;
    }

    public long fileSize() {
        return fileSize;
    }

    public boolean payloadIncluded() {
        return manifest.payloadIncluded();
    }

    public boolean runtimeLoadable() {
        return manifest.runtimeLoadable();
    }

    public String runtimeLoadMode() {
        return manifest.runtimeLoadMode();
    }

    public Map<String, Object> requireMap(String key) throws IOException {
        return manifest.requireMap(key);
    }

    public List<Map<String, Object>> requireListOfMaps(String key) throws IOException {
        return manifest.requireListOfMaps(key);
    }

    public RuntimeTensorCatalog runtimeTensorCatalog() throws IOException {
        return WdmlPackReader.mapPayloadTensors(this);
    }

    public Map<String, RuntimeTensor> mapPayloadTensors() throws IOException {
        return runtimeTensorCatalog().asMap();
    }

    public TensorCatalog buildTensorCatalog() throws IOException {
        return manifest.buildTensorCatalog();
    }

    public Path resolveSourcePath(Path modelDir, String requestedModelFileName) throws IOException {
        return manifest.resolveSourcePath(modelDir, requestedModelFileName);
    }

    public void validateRuntimeLoadable() throws IOException {
        manifest.validateRuntimeLoadable();
    }

    public void validateSourceFingerprint(Path source) throws IOException {
        Map<String, Object> sourceInfo = requireMap("source");
        String expectedFingerprint = stringValue(sourceInfo.get("fingerprint"));
        if (expectedFingerprint.isBlank()) {
            throw new IOException("Stale wdmlpack cache: missing source fingerprint");
        }
        SourceFingerprint actual = SourceFingerprint.read(source);
        if (!actual.matches(expectedFingerprint)) {
            throw new IOException("Stale wdmlpack cache: source fingerprint mismatch for "
                    + (source == null ? "<missing>" : source.getFileName()));
        }
    }

    public void validateSourceSize(Path source) throws IOException {
        if (!Files.isRegularFile(source)) {
            throw new IOException("wdmlpack source is missing: " + source);
        }
        Map<String, Object> sourceInfo = requireMap("source");
        long expectedSize = longValue(sourceInfo.get("sizeBytes"), -1L);
        long actualSize = Files.size(source);
        if (expectedSize >= 0 && expectedSize != actualSize) {
            throw new IOException("wdmlpack source size mismatch for " + source.getFileName()
                    + ": manifest=" + expectedSize + ", actual=" + actualSize);
        }
    }

    public static int intValue(Object value, int defaultValue) {
        return WdmlPackManifest.intValue(value, defaultValue);
    }

    public static long longValue(Object value, long defaultValue) {
        return WdmlPackManifest.longValue(value, defaultValue);
    }

    public static String stringValue(Object value) {
        return WdmlPackManifest.stringValue(value);
    }

    public static List<String> stringList(Object value) {
        return WdmlPackManifest.stringList(value);
    }

    public static long[] dimsValue(Object value) {
        return WdmlPackManifest.dimsValue(value);
    }

    public static Map<String, Object> castMap(Map<?, ?> raw) {
        return WdmlPackManifest.castMap(raw);
    }

    @Override
    public String toString() {
        return "RuntimeModelPackage{" + packagePath.getFileName()
                + ", mode=" + manifest.get("mode")
                + ", payloadIncluded=" + payloadIncluded()
                + '}';
    }
}
