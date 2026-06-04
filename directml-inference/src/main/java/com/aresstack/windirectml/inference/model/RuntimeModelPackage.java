package com.aresstack.windirectml.inference.model;

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
 * Format-neutral reader for internal {@code .wdmlpack} runtime packages.
 *
 * <p>Model-specific sources use this class to validate the container, read the
 * JSON manifest, map tensor payloads, and build a {@link TensorCatalog}. The
 * execution runtime therefore no longer needs to know whether a package was
 * originally compiled from ONNX, SafeTensors, or another import format.</p>
 */
public final class RuntimeModelPackage {

    private final Path packagePath;
    private final WdmlPackWriter.Header header;
    private final Map<String, Object> manifest;
    private final long fileSize;

    private RuntimeModelPackage(Path packagePath,
                                WdmlPackWriter.Header header,
                                Map<String, Object> manifest,
                                long fileSize) {
        this.packagePath = packagePath;
        this.header = header;
        this.manifest = manifest;
        this.fileSize = fileSize;
    }

    public static RuntimeModelPackage open(Path packagePath) throws IOException {
        Path normalized = Objects.requireNonNull(packagePath, "packagePath").toAbsolutePath().normalize();
        WdmlPackWriter.Header header = WdmlPackWriter.readHeader(normalized);
        Map<String, Object> manifest = WdmlPackWriter.readManifest(normalized);
        RuntimeModelPackage modelPackage = new RuntimeModelPackage(normalized, header, manifest, Files.size(normalized));
        modelPackage.validateRoot();
        return modelPackage;
    }

    public Path packagePath() {
        return packagePath;
    }

    public WdmlPackWriter.Header header() {
        return header;
    }

    public Map<String, Object> manifest() {
        return manifest;
    }

    public boolean payloadIncluded() {
        return Boolean.TRUE.equals(manifest.get("payloadIncluded"));
    }

    public boolean runtimeLoadable() {
        Object value = manifest.get("runtimeLoadable");
        return !(value instanceof Boolean) || Boolean.TRUE.equals(value);
    }

    public String runtimeLoadMode() {
        return stringValue(manifest.get("runtimeLoadMode"));
    }

    public Map<String, Object> requireMap(String key) throws IOException {
        Object value = manifest.get(key);
        if (!(value instanceof Map<?, ?> raw)) {
            throw new IOException("Invalid wdmlpack: missing " + key + " metadata");
        }
        return castMap(raw);
    }

    public List<Map<String, Object>> requireListOfMaps(String key) throws IOException {
        Object value = manifest.get(key);
        if (!(value instanceof List<?> rawList)) {
            throw new IOException("Invalid wdmlpack: missing " + key + " directory");
        }
        List<Map<String, Object>> out = new ArrayList<>();
        for (Object item : rawList) {
            if (item instanceof Map<?, ?> rawMap) {
                out.add(castMap(rawMap));
            }
        }
        return out;
    }

    public Map<String, RuntimeTensor> mapPayloadTensors() throws IOException {
        if (!payloadIncluded()) {
            return Map.of();
        }
        if (fileSize > Integer.MAX_VALUE) {
            throw new IOException("wdmlpack package is too large for the current mmap reader: " + packagePath
                    + " (" + fileSize + " bytes)");
        }
        try (FileChannel channel = FileChannel.open(packagePath, StandardOpenOption.READ)) {
            MappedByteBuffer mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);
            mapped.order(ByteOrder.LITTLE_ENDIAN);
            return mapPayloadTensors(mapped);
        }
    }

    public TensorCatalog buildTensorCatalog() throws IOException {
        List<TensorEntry> entries = new ArrayList<>();
        for (Map<String, Object> tensor : requireListOfMaps("tensors")) {
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

    public Path resolveSourcePath(Path modelDir, String requestedModelFileName) throws IOException {
        Map<String, Object> source = requireMap("source");
        String relativePath = stringValue(source.get("relativePath"));
        String fileName = stringValue(source.get("fileName"));
        String candidate = !relativePath.isBlank() ? relativePath : (!fileName.isBlank() ? fileName : requestedModelFileName);
        Path root = Objects.requireNonNull(modelDir, "modelDir").toAbsolutePath().normalize();
        Path resolved = root.resolve(candidate).toAbsolutePath().normalize();
        if (!resolved.startsWith(root)) {
            throw new IOException("Invalid wdmlpack source path escapes model directory: " + candidate);
        }
        return resolved;
    }

    public void validateRuntimeLoadable() throws IOException {
        if (!runtimeLoadable()) {
            throw new IOException("wdmlpack package is import-only and not runtime-loadable yet: " + runtimeLoadMode());
        }
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

    public static long longValue(Object value, long defaultValue) {
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

    public static String stringValue(Object value) {
        return value instanceof String s ? s : "";
    }

    public static List<String> stringList(Object value) {
        if (!(value instanceof List<?> list)) {
            return List.of();
        }
        List<String> out = new ArrayList<>(list.size());
        for (Object item : list) {
            out.add(String.valueOf(item));
        }
        return out;
    }

    public static long[] dimsValue(Object value) {
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

    @SuppressWarnings("unchecked")
    public static Map<String, Object> castMap(Map<?, ?> raw) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : raw.entrySet()) {
            out.put(String.valueOf(entry.getKey()), entry.getValue());
        }
        return out;
    }

    private Map<String, RuntimeTensor> mapPayloadTensors(MappedByteBuffer mapped) throws IOException {
        Map<String, RuntimeTensor> tensors = new LinkedHashMap<>();
        for (Map<String, Object> tensor : requireListOfMaps("tensors")) {
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
                RuntimeTensor runtimeTensor = mapTensorPayload(mapped, name, dims, dataType, payloadOffset, payloadLength);
                tensors.put(name, runtimeTensor);
            } else {
                tensors.put(name, new RuntimeTensor(name, dims, dataType, ByteBuffer.allocate(0), 0));
            }
        }
        return tensors;
    }

    private RuntimeTensor mapTensorPayload(MappedByteBuffer mapped,
                                           String name,
                                           long[] dims,
                                           int dataType,
                                           long payloadOffset,
                                           long payloadLength) throws IOException {
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
        return new RuntimeTensor(name, dims, dataType, raw, (int) payloadLength);
    }

    private void validateRoot() throws IOException {
        if (!"wdmlpack".equals(manifest.get("format"))) {
            throw new IOException("Invalid wdmlpack manifest format: " + manifest.get("format"));
        }
        int version = intValue(manifest.get("version"), -1);
        if (version != WdmlPackWriter.VERSION) {
            throw new IOException("Unsupported wdmlpack manifest version: " + version);
        }
        boolean payloadIncluded = payloadIncluded();
        if (payloadIncluded && !header.payloadIncluded()) {
            throw new IOException("wdmlpack manifest says payloadIncluded=true but the container header has no payload");
        }
        if (!payloadIncluded && header.payloadIncluded()) {
            throw new IOException("wdmlpack container has payload but manifest says payloadIncluded=false");
        }
    }

    @Override
    public String toString() {
        return "RuntimeModelPackage{" + packagePath.getFileName()
                + ", mode=" + manifest.get("mode")
                + ", payloadIncluded=" + payloadIncluded()
                + '}';
    }
}
