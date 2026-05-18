package com.aresstack.windirectml.encoder.safetensors;

import com.aresstack.windirectml.runtime.TensorDataType;
import com.aresstack.windirectml.runtime.TensorShape;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * Reader für das {@code .safetensors}-Format
 * (siehe https://github.com/huggingface/safetensors).
 *
 * <p>Datei-Layout:
 * <pre>
 *   [ 8 Bytes ]    u64 little-endian: Länge des JSON-Headers
 *   [ N Bytes ]    UTF-8-JSON: Tensor-Metadaten
 *   [ rest    ]    konkatenierte Tensor-Daten (Reihenfolge wie im Header)
 * </pre>
 *
 * <p>Der Reader lädt den Datenbereich per Memory-Mapping, hält selbst keine
 * Tensor-Kopien und ist daher leichtgewichtig. Tensor-Daten werden als
 * read-only {@link ByteBuffer}-Slice ausgeliefert.
 */
public final class SafetensorsReader implements AutoCloseable {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final long MAX_HEADER_BYTES = 100L * 1024 * 1024;

    private final FileChannel channel;
    private final MappedByteBuffer mapping;
    private final long dataSectionOffset;
    private final Map<String, SafetensorsEntry> entries;

    private SafetensorsReader(FileChannel channel, MappedByteBuffer mapping,
                              long dataSectionOffset, Map<String, SafetensorsEntry> entries) {
        this.channel = channel;
        this.mapping = mapping;
        this.dataSectionOffset = dataSectionOffset;
        this.entries = entries;
    }

    public static SafetensorsReader open(Path file) throws SafetensorsException {
        FileChannel channel = null;
        try {
            channel = FileChannel.open(file, StandardOpenOption.READ);
            long fileSize = channel.size();
            if (fileSize < 8) throw new SafetensorsException("file too small: " + fileSize);

            ByteBuffer headerLenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            if (channel.read(headerLenBuf, 0) != 8) {
                throw new SafetensorsException("could not read 8-byte header length");
            }
            headerLenBuf.flip();
            long headerLen = headerLenBuf.getLong();
            if (headerLen <= 0 || headerLen > MAX_HEADER_BYTES || 8 + headerLen > fileSize) {
                throw new SafetensorsException("invalid header length: " + headerLen);
            }

            ByteBuffer headerBuf = ByteBuffer.allocate((int) headerLen);
            if (channel.read(headerBuf, 8) != headerLen) {
                throw new SafetensorsException("could not read JSON header");
            }
            String headerJson = new String(headerBuf.array(), StandardCharsets.UTF_8);
            JsonNode header = MAPPER.readTree(headerJson);

            long dataSectionOffset = 8 + headerLen;
            long dataSectionLen = fileSize - dataSectionOffset;

            Map<String, SafetensorsEntry> entries = parseHeader(header, dataSectionLen);

            MappedByteBuffer mapping = channel.map(FileChannel.MapMode.READ_ONLY,
                    dataSectionOffset, dataSectionLen);
            mapping.order(ByteOrder.LITTLE_ENDIAN);

            FileChannel out = channel;
            channel = null;
            return new SafetensorsReader(out, mapping, dataSectionOffset, entries);
        } catch (IOException e) {
            throw new SafetensorsException("failed to open " + file, e);
        } finally {
            if (channel != null) {
                try { channel.close(); } catch (IOException ignored) {}
            }
        }
    }

    private static Map<String, SafetensorsEntry> parseHeader(JsonNode header, long dataSectionLen)
            throws SafetensorsException {
        if (!header.isObject()) throw new SafetensorsException("header is not a JSON object");
        Map<String, SafetensorsEntry> map = new LinkedHashMap<>();
        Iterator<Map.Entry<String, JsonNode>> it = header.fields();
        while (it.hasNext()) {
            Map.Entry<String, JsonNode> e = it.next();
            String name = e.getKey();
            if ("__metadata__".equals(name)) continue;

            JsonNode node = e.getValue();
            JsonNode dtypeNode = node.get("dtype");
            JsonNode shapeNode = node.get("shape");
            JsonNode offsetsNode = node.get("data_offsets");
            if (dtypeNode == null || shapeNode == null || offsetsNode == null
                    || !shapeNode.isArray() || !offsetsNode.isArray() || offsetsNode.size() != 2) {
                throw new SafetensorsException("invalid tensor entry: " + name);
            }
            TensorDataType dataType = mapDataType(dtypeNode.asText(), name);
            int[] dims = new int[shapeNode.size()];
            if (dims.length == 0) throw new SafetensorsException("empty shape for tensor " + name);
            for (int i = 0; i < dims.length; i++) {
                long d = shapeNode.get(i).asLong();
                if (d <= 0 || d > Integer.MAX_VALUE) {
                    throw new SafetensorsException("dimension out of range for " + name + ": " + d);
                }
                dims[i] = (int) d;
            }
            long start = offsetsNode.get(0).asLong();
            long end = offsetsNode.get(1).asLong();
            if (start < 0 || end < start || end > dataSectionLen) {
                throw new SafetensorsException("data_offsets out of range for " + name
                        + ": [" + start + "," + end + "), section=" + dataSectionLen);
            }
            map.put(name, new SafetensorsEntry(
                    name, dataType, TensorShape.of(dims), start, end - start));
        }
        return map;
    }

    private static TensorDataType mapDataType(String dtype, String tensorName) throws SafetensorsException {
        return switch (dtype) {
            case "F32"  -> TensorDataType.FLOAT32;
            case "F16"  -> TensorDataType.FLOAT16;
            case "BF16" -> TensorDataType.BFLOAT16;
            case "I32"  -> TensorDataType.INT32;
            case "I8"   -> TensorDataType.INT8;
            case "U8"   -> TensorDataType.UINT8;
            default -> throw new SafetensorsException(
                    "unsupported dtype '" + dtype + "' for tensor " + tensorName);
        };
    }

    // ── Public API ────────────────────────────────────────────────────────

    public Set<String> tensorNames() {
        return entries.keySet();
    }

    public SafetensorsEntry entry(String name) throws SafetensorsException {
        SafetensorsEntry e = entries.get(name);
        if (e == null) throw new SafetensorsException("tensor not found: " + name);
        return e;
    }

    /**
     * Liefert einen read-only {@link ByteBuffer}-Slice mit den Rohdaten des
     * Tensors. Der Slice teilt den Speicher mit der Memory-Map – keine Kopie.
     */
    public ByteBuffer rawData(String name) throws SafetensorsException {
        SafetensorsEntry e = entry(name);
        ByteBuffer slice = mapping.slice((int) e.dataOffset(), (int) e.byteLength())
                .asReadOnlyBuffer();
        slice.order(ByteOrder.LITTLE_ENDIAN);
        return slice;
    }

    /** Lade einen FLOAT32-Tensor vollständig in ein Java-Array (Kopie). */
    public float[] readFloat32(String name) throws SafetensorsException {
        SafetensorsEntry e = entry(name);
        if (e.dataType() != TensorDataType.FLOAT32) {
            throw new SafetensorsException("tensor " + name + " is not F32 but " + e.dataType());
        }
        int n = Math.toIntExact(e.shape().elementCount());
        float[] out = new float[n];
        rawData(name).asFloatBuffer().get(out);
        return out;
    }

    public long dataSectionOffset() {
        return dataSectionOffset;
    }

    @Override
    public void close() {
        try {
            channel.close();
        } catch (IOException ignored) {
            // mapped buffer remains until GC; nothing actionable here
        }
    }
}

