package com.aresstack.windirectml.inference.model;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Parsed JSON manifest of an internal {@code .wdmlpack} package.
 *
 * <p>This class is intentionally format-neutral. It exposes typed accessors for
 * package-level metadata and keeps JSON map casting out of model-family runtime
 * sources.</p>
 */
public final class WdmlPackManifest {

    private final Map<String, Object> root;

    public WdmlPackManifest(Map<String, Object> root) {
        Objects.requireNonNull(root, "root");
        this.root = Map.copyOf(root);
    }

    public Map<String, Object> root() {
        return root;
    }

    public Object get(String key) {
        return root.get(key);
    }

    public boolean payloadIncluded() {
        return Boolean.TRUE.equals(root.get("payloadIncluded"));
    }

    public boolean runtimeLoadable() {
        Object value = root.get("runtimeLoadable");
        return !(value instanceof Boolean) || Boolean.TRUE.equals(value);
    }

    public String runtimeLoadMode() {
        return stringValue(root.get("runtimeLoadMode"));
    }

    public Map<String, Object> requireMap(String key) throws IOException {
        Object value = root.get(key);
        if (!(value instanceof Map<?, ?> raw)) {
            throw new IOException("Invalid wdmlpack: missing " + key + " metadata");
        }
        return castMap(raw);
    }

    public List<Map<String, Object>> requireListOfMaps(String key) throws IOException {
        Object value = root.get(key);
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
        Path rootPath = Objects.requireNonNull(modelDir, "modelDir").toAbsolutePath().normalize();
        Path resolved = rootPath.resolve(candidate).toAbsolutePath().normalize();
        if (!resolved.startsWith(rootPath)) {
            throw new IOException("Invalid wdmlpack source path escapes model directory: " + candidate);
        }
        return resolved;
    }

    public void validateRuntimeLoadable() throws IOException {
        if (!runtimeLoadable()) {
            throw new IOException("wdmlpack package is import-only and not runtime-loadable yet: " + runtimeLoadMode());
        }
    }

    public void validateRoot(WdmlPackWriter.Header header) throws IOException {
        if (!"wdmlpack".equals(root.get("format"))) {
            throw new IOException("Invalid wdmlpack manifest format: " + root.get("format"));
        }
        int version = intValue(root.get("version"), -1);
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
            if (item instanceof Number n) {
                dims[i] = n.longValue();
            } else if (item instanceof String s && !s.isBlank()) {
                try {
                    dims[i] = Long.parseLong(s);
                } catch (NumberFormatException ignored) {
                    dims[i] = 0L;
                }
            }
        }
        return dims;
    }

    public static Map<String, Object> castMap(Map<?, ?> raw) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : raw.entrySet()) {
            out.put(String.valueOf(entry.getKey()), entry.getValue());
        }
        return out;
    }
}
