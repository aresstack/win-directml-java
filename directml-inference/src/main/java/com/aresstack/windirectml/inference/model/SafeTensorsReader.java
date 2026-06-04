package com.aresstack.windirectml.inference.model;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
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
 * Minimal SafeTensors reader used by the model-import layer.
 *
 * <p>The reader intentionally parses only the file container and tensor
 * directory. It does not attempt to infer a neural-network graph. Model-specific
 * compilers, such as Qwen, use this to build a {@link TensorCatalog} and later
 * transform the source tensors into a runtime-native {@code .wdmlpack} layout.</p>
 */
public final class SafeTensorsReader {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int HEADER_LENGTH_BYTES = Long.BYTES;

    private SafeTensorsReader() {
    }

    public record SafeTensorEntry(
            String name,
            String dtype,
            int onnxDataType,
            long[] shape,
            long dataBegin,
            long dataEnd,
            long byteLength,
            ByteBuffer data
    ) {
        public SafeTensorEntry {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(dtype, "dtype");
            shape = shape != null ? shape.clone() : new long[0];
            data = data != null ? data.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN) : null;
        }

        @Override
        public long[] shape() {
            return shape.clone();
        }

        public ByteBuffer dataBuffer() {
            ByteBuffer duplicate = data.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
            duplicate.position(0);
            duplicate.limit(Math.toIntExact(byteLength));
            return duplicate;
        }
    }

    public record SafeTensorsFile(Path path,
                                  Map<String, SafeTensorEntry> tensors,
                                  Map<String, String> metadata,
                                  long dataBaseOffset,
                                  long fileSize) {
        public SafeTensorsFile {
            Objects.requireNonNull(path, "path");
            tensors = Map.copyOf(new LinkedHashMap<>(tensors));
            metadata = Map.copyOf(new LinkedHashMap<>(metadata));
        }
    }

    public static SafeTensorsFile read(Path file) throws IOException {
        Objects.requireNonNull(file, "file");
        if (!Files.isRegularFile(file)) {
            throw new IOException("SafeTensors file not found: " + file);
        }
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
            long fileSize = channel.size();
            if (fileSize < HEADER_LENGTH_BYTES) {
                throw new IOException("Invalid SafeTensors file: too small: " + file);
            }
            if (fileSize > Integer.MAX_VALUE) {
                throw new IOException("SafeTensors file is too large for the current mmap reader: "
                        + file.getFileName() + " (" + fileSize + " bytes)");
            }
            MappedByteBuffer mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);
            mapped.order(ByteOrder.LITTLE_ENDIAN);
            long headerLength = mapped.getLong(0);
            validateHeaderLength(file, fileSize, headerLength);

            ByteBuffer headerBuffer = mapped.asReadOnlyBuffer();
            headerBuffer.position(HEADER_LENGTH_BYTES);
            headerBuffer.limit(Math.toIntExact(HEADER_LENGTH_BYTES + headerLength));
            byte[] headerBytes = new byte[Math.toIntExact(headerLength)];
            headerBuffer.get(headerBytes);

            JsonNode root = MAPPER.readTree(new String(headerBytes, StandardCharsets.UTF_8));
            long dataBaseOffset = HEADER_LENGTH_BYTES + headerLength;
            Map<String, String> metadata = parseMetadata(root.get("__metadata__"));
            Map<String, SafeTensorEntry> tensors = new LinkedHashMap<>();

            var fields = root.fields();
            while (fields.hasNext()) {
                Map.Entry<String, JsonNode> field = fields.next();
                String name = field.getKey();
                if ("__metadata__".equals(name)) {
                    continue;
                }
                tensors.put(name, parseTensor(file, mapped, fileSize, dataBaseOffset, name, field.getValue()));
            }
            return new SafeTensorsFile(file.toAbsolutePath().normalize(), tensors, metadata, dataBaseOffset, fileSize);
        }
    }

    private static SafeTensorEntry parseTensor(Path file,
                                               MappedByteBuffer mapped,
                                               long fileSize,
                                               long dataBaseOffset,
                                               String name,
                                               JsonNode node) throws IOException {
        if (!node.isObject()) {
            throw new IOException("Invalid SafeTensors tensor entry for " + name + " in " + file.getFileName());
        }
        String dtype = requiredText(node, "dtype", name, file).toUpperCase(Locale.ROOT);
        long[] shape = requiredLongArray(node, "shape", name, file);
        long[] offsets = requiredLongArray(node, "data_offsets", name, file);
        if (offsets.length != 2) {
            throw new IOException("Invalid SafeTensors data_offsets for " + name + " in " + file.getFileName());
        }
        long begin = offsets[0];
        long end = offsets[1];
        if (begin < 0 || end < begin) {
            throw new IOException("Invalid SafeTensors data range for " + name + ": [" + begin + ", " + end + "]");
        }
        long absoluteBegin = dataBaseOffset + begin;
        long absoluteEnd = dataBaseOffset + end;
        if (absoluteBegin < dataBaseOffset || absoluteEnd < absoluteBegin || absoluteEnd > fileSize) {
            throw new IOException("SafeTensors data range for " + name + " escapes file bounds: ["
                    + begin + ", " + end + "] fileSize=" + fileSize);
        }
        long byteLength = end - begin;
        validateByteLength(dtype, shape, byteLength, name, file);

        ByteBuffer data = mapped.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
        data.position(Math.toIntExact(absoluteBegin));
        data.limit(Math.toIntExact(absoluteEnd));
        ByteBuffer slice = data.slice().order(ByteOrder.LITTLE_ENDIAN).asReadOnlyBuffer();
        return new SafeTensorEntry(name, dtype, toOnnxDataType(dtype), shape, begin, end, byteLength, slice);
    }

    private static void validateHeaderLength(Path file, long fileSize, long headerLength) throws IOException {
        if (headerLength < 2 || headerLength > fileSize - HEADER_LENGTH_BYTES) {
            throw new IOException("Invalid SafeTensors header length for " + file.getFileName() + ": " + headerLength);
        }
        if (HEADER_LENGTH_BYTES + headerLength > Integer.MAX_VALUE) {
            throw new IOException("SafeTensors header too large for current reader: " + headerLength);
        }
    }

    private static Map<String, String> parseMetadata(JsonNode metadataNode) {
        Map<String, String> metadata = new LinkedHashMap<>();
        if (metadataNode == null || !metadataNode.isObject()) {
            return metadata;
        }
        var fields = metadataNode.fields();
        while (fields.hasNext()) {
            Map.Entry<String, JsonNode> field = fields.next();
            metadata.put(field.getKey(), field.getValue().asText());
        }
        return metadata;
    }

    private static String requiredText(JsonNode node, String field, String tensorName, Path file) throws IOException {
        JsonNode value = node.get(field);
        if (value == null || !value.isTextual()) {
            throw new IOException("Invalid SafeTensors tensor " + tensorName + " in " + file.getFileName()
                    + ": missing textual " + field);
        }
        return value.asText();
    }

    private static long[] requiredLongArray(JsonNode node, String field, String tensorName, Path file) throws IOException {
        JsonNode value = node.get(field);
        if (value == null || !value.isArray()) {
            throw new IOException("Invalid SafeTensors tensor " + tensorName + " in " + file.getFileName()
                    + ": missing array " + field);
        }
        List<Long> values = new ArrayList<>();
        for (JsonNode item : value) {
            if (!item.canConvertToLong()) {
                throw new IOException("Invalid SafeTensors tensor " + tensorName + " in " + file.getFileName()
                        + ": non-integer " + field);
            }
            values.add(item.asLong());
        }
        long[] out = new long[values.size()];
        for (int i = 0; i < values.size(); i++) {
            out[i] = values.get(i);
        }
        return out;
    }

    private static void validateByteLength(String dtype, long[] shape, long byteLength, String tensorName, Path file)
            throws IOException {
        long elements = 1;
        for (long dim : shape) {
            if (dim < 0) {
                throw new IOException("Invalid negative dimension for SafeTensors tensor " + tensorName + " in "
                        + file.getFileName() + ": " + dim);
            }
            elements = Math.multiplyExact(elements, dim);
        }
        long expected = Math.multiplyExact(elements, bytesPerElement(dtype));
        if (expected != byteLength) {
            throw new IOException("SafeTensors byte length mismatch for " + tensorName + " in " + file.getFileName()
                    + ": expected " + expected + " bytes from dtype/shape, got " + byteLength);
        }
    }

    public static int bytesPerElement(String dtype) throws IOException {
        return switch (dtype.toUpperCase(Locale.ROOT)) {
            case "F16", "BF16", "I16", "U16" -> 2;
            case "F32", "I32", "U32" -> 4;
            case "F64", "I64", "U64" -> 8;
            case "I8", "U8", "BOOL" -> 1;
            default -> throw new IOException("Unsupported SafeTensors dtype: " + dtype);
        };
    }

    public static int toOnnxDataType(String dtype) throws IOException {
        return switch (dtype.toUpperCase(Locale.ROOT)) {
            case "F32" -> 1;
            case "U8" -> 2;
            case "I8" -> 3;
            case "I32" -> 6;
            case "I64" -> 7;
            case "F16" -> 10;
            case "BF16" -> 16;
            case "BOOL" -> 9;
            case "F64" -> 11;
            case "I16" -> 5;
            case "U16" -> 4;
            case "U32" -> 12;
            case "U64" -> 13;
            default -> throw new IOException("Unsupported SafeTensors dtype: " + dtype);
        };
    }
}
