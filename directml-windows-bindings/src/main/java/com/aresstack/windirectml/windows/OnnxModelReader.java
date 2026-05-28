package com.aresstack.windirectml.windows;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Minimal ONNX protobuf reader – parses what is needed for MNIST-family models
 * (validated with {@code mnist-12.onnx} and {@code mnist-8.onnx}).
 * <p>
 * No dependency on Google Protobuf or any ONNX library.
 * Hand-written decoder for the ONNX wire format (protobuf3).
 * <p>
 * Only supports: ModelProto → GraphProto → NodeProto, TensorProto (float32 / raw_data).
 */
public final class OnnxModelReader {

    private static final Logger log = LoggerFactory.getLogger(OnnxModelReader.class);

    private OnnxModelReader() {}

    // ── Public data types ────────────────────────────────────────────────

    /** ONNX data type constants. */
    public static final int ONNX_FLOAT = 1, ONNX_UINT8 = 2, ONNX_INT8 = 3,
            ONNX_INT32 = 6, ONNX_INT64 = 7, ONNX_FLOAT16 = 10;

    /**
     * Parsed ONNX tensor. {@code data} holds float values (for FLOAT type or
     * after conversion). {@code rawBytes} holds the raw byte payload for
     * non-FLOAT types (INT8, UINT8, INT32). Both may be populated.
     */
    public record OnnxTensor(String name, long[] dims, int dataType,
                              float[] data, byte[] rawBytes) {
        /** Convenience constructor for float-only tensors (backward compat). */
        public OnnxTensor(String name, long[] dims, int dataType, float[] data) {
            this(name, dims, dataType, data, new byte[0]);
        }

        public int elementCount() {
            int n = 1;
            for (long d : dims) n *= (int) d;
            return n;
        }

        /** Read a single signed int8 value from rawBytes. */
        public byte getInt8(int index) { return rawBytes[index]; }

        /** Read a single unsigned uint8 value from rawBytes. */
        public int getUint8(int index) { return rawBytes[index] & 0xFF; }

        /** Read a little-endian int32 from rawBytes at element index. */
        public int getInt32(int index) {
            int off = index * 4;
            return (rawBytes[off] & 0xFF)
                    | ((rawBytes[off + 1] & 0xFF) << 8)
                    | ((rawBytes[off + 2] & 0xFF) << 16)
                    | ((rawBytes[off + 3] & 0xFF) << 24);
        }

        /** Read a single float from the data array, or 0 if out of bounds. */
        public float getFloat(int index) {
            return index < data.length ? data[index] : 0f;
        }
    }

    public record OnnxNode(String opType, List<String> inputs, List<String> outputs,
                            Map<String, Object> attrs) {
        @SuppressWarnings("unchecked")
        public List<Long> getInts(String name) {
            Object v = attrs.get(name);
            return v instanceof List<?> ? (List<Long>) v : List.of();
        }
        public long getInt(String name, long def) {
            Object v = attrs.get(name);
            return v instanceof Long l ? l : def;
        }
        public String getString(String name, String def) {
            Object v = attrs.get(name);
            return v instanceof String s ? s : def;
        }
    }

    public record OnnxGraph(String name, List<OnnxNode> nodes,
                             Map<String, OnnxTensor> initializers,
                             List<String> inputNames, List<String> outputNames) {}

    // ── Entry point ──────────────────────────────────────────────────────

    public static OnnxGraph parse(Path onnxFile) throws IOException {
        byte[] bytes = Files.readAllBytes(onnxFile);
        ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        log.info("Parsing ONNX model: {} ({} bytes)", onnxFile.getFileName(), bytes.length);

        OnnxGraph[] graph = new OnnxGraph[1];

        try {
            // ModelProto: field 7 = graph (GraphProto)
            while (buf.hasRemaining()) {
                int loopStart = buf.position();
                int tag = readVarint32(buf);
                int fieldNum = tag >>> 3;
                int wireType = tag & 0x7;
                if (fieldNum == 7 && wireType == 2) {
                    int len = readVarint32(buf);
                    // Clamp to buffer limit: ONNX Community exports include external
                    // raw_data stubs in the outer length, inflating it past EOF.
                    int end = Math.min(buf.position() + len, buf.limit());
                    graph[0] = parseGraph(buf, end);
                    buf.position(end);
                } else {
                    skipField(buf, wireType);
                }
                // Safety: if no progress was made (e.g. corrupt tag at EOF), abort
                // to avoid an infinite loop. Without this guard, readVarint32 on an
                // empty buffer returns 0 and skipField(_, 0) also makes no progress.
                if (buf.position() == loopStart) break;
            }
        } catch (RuntimeException e) {
            // Wrap underflow / illegal-argument errors with the byte offset so the
            // caller (e.g. describeUnsupportedFormat) can produce a useful message
            // instead of a bare "BufferUnderflowException (no detail message)".
            throw new IOException("Failed to parse ONNX model '" + onnxFile.getFileName()
                    + "' at byte offset " + buf.position() + " of " + bytes.length
                    + " (" + e.getClass().getSimpleName()
                    + (e.getMessage() != null ? ": " + e.getMessage() : "") + ")", e);
        }

        if (graph[0] == null) throw new IOException("No graph found in ONNX model");
        log.info("Parsed ONNX graph '{}': {} nodes, {} initializers",
                graph[0].name, graph[0].nodes.size(), graph[0].initializers.size());
        return graph[0];
    }

    // ── GraphProto ───────────────────────────────────────────────────────

    private static OnnxGraph parseGraph(ByteBuffer buf, int end) {
        List<OnnxNode> nodes = new ArrayList<>();
        Map<String, OnnxTensor> initializers = new LinkedHashMap<>();
        List<String> inputNames = new ArrayList<>();
        List<String> outputNames = new ArrayList<>();
        String name = "";

        while (buf.position() < end) {
            int loopStart = buf.position();
            int tag = readVarint32(buf);
            int fieldNum = tag >>> 3;
            int wireType = tag & 0x7;
            switch (fieldNum) {
                case 1 -> { // repeated NodeProto node
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        int nodeEnd = Math.min(buf.position() + len, buf.limit());
                        nodes.add(parseNode(buf, nodeEnd));
                        buf.position(nodeEnd);
                    } else skipField(buf, wireType);
                }
                case 2 -> { // string name
                    if (wireType == 2) name = readString(buf);
                    else skipField(buf, wireType);
                }
                case 5 -> { // repeated TensorProto initializer
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        // Clamp to buffer limit: ONNX Community exports emit a raw_data
                        // stub whose declared length matches the external file size,
                        // which inflates the outer TensorProto length past EOF.
                        int tEnd = Math.min(buf.position() + len, buf.limit());
                        OnnxTensor t = parseTensor(buf, tEnd);
                        buf.position(tEnd);
                        if (t != null) initializers.put(t.name, t);
                    } else skipField(buf, wireType);
                }
                case 11 -> { // repeated ValueInfoProto input
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        int viEnd = Math.min(buf.position() + len, buf.limit());
                        String n = parseValueInfoName(buf, viEnd);
                        buf.position(viEnd);
                        if (n != null) inputNames.add(n);
                    } else skipField(buf, wireType);
                }
                case 12 -> { // repeated ValueInfoProto output
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        int viEnd = Math.min(buf.position() + len, buf.limit());
                        String n = parseValueInfoName(buf, viEnd);
                        buf.position(viEnd);
                        if (n != null) outputNames.add(n);
                    } else skipField(buf, wireType);
                }
                default -> skipField(buf, wireType);
            }
            // Safety: if no progress was made (corrupt tag / EOF / inflated end),
            // abort to prevent an infinite loop.
            if (buf.position() == loopStart) break;
        }
        return new OnnxGraph(name, nodes, initializers, inputNames, outputNames);
    }

    // ── NodeProto ────────────────────────────────────────────────────────

    private static OnnxNode parseNode(ByteBuffer buf, int end) {
        List<String> inputs = new ArrayList<>();
        List<String> outputs = new ArrayList<>();
        String name = "";
        String opType = "";
        Map<String, Object> attrs = new LinkedHashMap<>();

        while (buf.position() < end) {
            int loopStart = buf.position();
            int tag = readVarint32(buf);
            int fieldNum = tag >>> 3;
            int wireType = tag & 0x7;
            switch (fieldNum) {
                case 1 -> { if (wireType == 2) inputs.add(readString(buf)); else skipField(buf, wireType); }
                case 2 -> { if (wireType == 2) outputs.add(readString(buf)); else skipField(buf, wireType); }
                case 3 -> { if (wireType == 2) name = readString(buf); else skipField(buf, wireType); }
                case 4 -> { if (wireType == 2) opType = readString(buf); else skipField(buf, wireType); }
                case 5 -> { // AttributeProto
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        int aEnd = Math.min(buf.position() + len, buf.limit());
                        parseAttribute(buf, aEnd, attrs);
                        buf.position(aEnd);
                    } else skipField(buf, wireType);
                }
                default -> skipField(buf, wireType);
            }
            if (buf.position() == loopStart) break;
        }
        return new OnnxNode(opType, inputs, outputs, attrs);
    }

    // ── AttributeProto ───────────────────────────────────────────────────

    private static void parseAttribute(ByteBuffer buf, int end, Map<String, Object> attrs) {
        String attrName = "";
        Long intVal = null;
        Float floatVal = null;
        String strVal = null;
        List<Long> ints = new ArrayList<>();

        while (buf.position() < end) {
            int loopStart = buf.position();
            int tag = readVarint32(buf);
            int fieldNum = tag >>> 3;
            int wireType = tag & 0x7;
            switch (fieldNum) {
                case 1 -> { if (wireType == 2) attrName = readString(buf); else skipField(buf, wireType); }
                case 2 -> { if (wireType == 0) intVal = readVarint64(buf); else skipField(buf, wireType); }
                case 3 -> { if (wireType == 2) strVal = readString(buf); else skipField(buf, wireType); }
                case 4 -> { if (wireType == 5) floatVal = Float.intBitsToFloat(buf.getInt()); else skipField(buf, wireType); }
                case 7 -> { // repeated float — skip for now
                    skipField(buf, wireType);
                }
                case 8 -> { // repeated int64 ints (packed or repeated)
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        int pEnd = Math.min(buf.position() + len, buf.limit());
                        while (buf.position() < pEnd) {
                            int innerStart = buf.position();
                            ints.add(readVarint64(buf));
                            if (buf.position() == innerStart) break;
                        }
                    } else if (wireType == 0) {
                        ints.add(readVarint64(buf));
                    } else skipField(buf, wireType);
                }
                default -> skipField(buf, wireType);
            }
            if (buf.position() == loopStart) break;
        }

        if (!attrName.isEmpty()) {
            if (!ints.isEmpty()) attrs.put(attrName, ints);
            else if (intVal != null) attrs.put(attrName, intVal);
            else if (floatVal != null) attrs.put(attrName, floatVal);
            else if (strVal != null) attrs.put(attrName, strVal);
        }
    }

    // ── TensorProto ──────────────────────────────────────────────────────

    /**
     * TensorProto.DataLocation enum: DEFAULT = 0 (inline), EXTERNAL = 1.
     */
    private static final int DATA_LOCATION_EXTERNAL = 1;

    private static OnnxTensor parseTensor(ByteBuffer buf, int end) {
        List<Long> dims = new ArrayList<>();
        int dataType = 0;
        String name = "";
        float[] floatData = null;
        byte[] rawData = null;
        List<Integer> int32Values = null; // from field 5 (packed varints)
        int dataLocation = 0; // TensorProto.data_location (field 14)

        while (buf.position() < end) {
            int loopStart = buf.position();
            int tag = readVarint32(buf);
            int fieldNum = tag >>> 3;
            int wireType = tag & 0x7;
            switch (fieldNum) {
                case 1 -> { // repeated int64 dims (packed or repeated)
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        int pEnd = Math.min(buf.position() + len, buf.limit());
                        while (buf.position() < pEnd) {
                            int innerStart = buf.position();
                            dims.add(readVarint64(buf));
                            if (buf.position() == innerStart) break;
                        }
                    } else if (wireType == 0) {
                        dims.add(readVarint64(buf));
                    } else skipField(buf, wireType);
                }
                case 2 -> { // int32 data_type
                    if (wireType == 0) dataType = readVarint32(buf);
                    else skipField(buf, wireType);
                }
                case 4 -> { // repeated float float_data (packed fixed32)
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        int count = len / 4;
                        floatData = new float[count];
                        for (int i = 0; i < count; i++) floatData[i] = Float.intBitsToFloat(buf.getInt());
                    } else if (wireType == 5) {
                        floatData = new float[]{ Float.intBitsToFloat(buf.getInt()) };
                    } else skipField(buf, wireType);
                }
                case 5 -> { // repeated int32 int32_data (packed varints)
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        int pEnd = Math.min(buf.position() + len, buf.limit());
                        int32Values = new ArrayList<>();
                        while (buf.position() < pEnd) {
                            int innerStart = buf.position();
                            int32Values.add(readVarint32(buf));
                            if (buf.position() == innerStart) break;
                        }
                    } else if (wireType == 0) {
                        if (int32Values == null) int32Values = new ArrayList<>();
                        int32Values.add(readVarint32(buf));
                    } else skipField(buf, wireType);
                }
                case 8 -> { // string name
                    if (wireType == 2) name = readString(buf);
                    else skipField(buf, wireType);
                }
                case 9 -> { // raw_data (bytes) — inline raw tensor payload
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        if (len < 0 || Integer.toUnsignedLong(len) > buf.remaining()) {
                            // Raw data length exceeds the available buffer — the tensor data
                            // cannot be read inline. This occurs in ONNX Community models where
                            // lm_head (or other large tensors) is stored in the external data
                            // file but its TensorProto still carries a raw_data stub whose
                            // encoded length exceeds the model.onnx file size.
                            // Recovery: skip to the TensorProto boundary so parsing can
                            // continue normally; the tensor will be resolved from externalRefs.
                            log.warn("Tensor '{}' raw_data length {} exceeds remaining buffer {} "
                                            + "— skipping inline data (will be resolved from external refs)",
                                    name, Integer.toUnsignedString(len), buf.remaining());
                            // Jump to end of this TensorProto, clamped to buffer limit
                            // (external-data stubs can also inflate the outer message length).
                            buf.position(Math.min(end, buf.limit()));
                        } else {
                            rawData = new byte[len];
                            buf.get(rawData);
                        }
                    } else skipField(buf, wireType);
                }
                case 13 -> { // external_data (repeated StringStringEntryProto) — skip, parsed elsewhere
                    skipField(buf, wireType);
                }
                case 14 -> { // data_location (DataLocation enum)
                    if (wireType == 0) dataLocation = readVarint32(buf);
                    else skipField(buf, wireType);
                }
                default -> skipField(buf, wireType);
            }
            if (buf.position() == loopStart) break;
        }

        // Build the final float[] and byte[] based on data type.
        // For external tensors (data_location == EXTERNAL), the inline payload
        // is empty: weights live in model.onnx.data (or model.onnx_data) and are resolved separately.
        float[] data;
        byte[] rawBytes;

        if (dataLocation == DATA_LOCATION_EXTERNAL) {
            data = new float[0];
            rawBytes = new byte[0];
        } else if (floatData != null) {
            // FLOAT tensor with explicit float_data
            data = floatData;
            rawBytes = new byte[0];
        } else if (rawData != null && dataType == ONNX_FLOAT) {
            // FLOAT tensor with raw_data
            ByteBuffer rb = ByteBuffer.wrap(rawData).order(ByteOrder.LITTLE_ENDIAN);
            data = new float[rawData.length / 4];
            for (int i = 0; i < data.length; i++) data[i] = rb.getFloat();
            rawBytes = new byte[0];
        } else if (rawData != null) {
            // Non-FLOAT tensor with raw byte data (INT8, UINT8, INT32)
            data = new float[0];
            rawBytes = rawData;
        } else if (int32Values != null) {
            // Non-FLOAT tensor with int32_data (varint-encoded)
            data = new float[0];
            if (dataType == ONNX_INT8 || dataType == ONNX_UINT8) {
                // Store each int32 value as a single byte
                rawBytes = new byte[int32Values.size()];
                for (int i = 0; i < int32Values.size(); i++)
                    rawBytes[i] = int32Values.get(i).byteValue();
            } else if (dataType == ONNX_INT32) {
                // Store as little-endian int32 bytes
                rawBytes = new byte[int32Values.size() * 4];
                ByteBuffer bb = ByteBuffer.wrap(rawBytes).order(ByteOrder.LITTLE_ENDIAN);
                for (int v : int32Values) bb.putInt(v);
            } else if (dataType == ONNX_FLOAT16) {
                // FP16: ONNX stores fp16 values in int32_data.
                // Each int32 contains ONE fp16 value in its lower 16 bits.
                // Convert to raw little-endian fp16 bytes (2 bytes per value).
                rawBytes = new byte[int32Values.size() * 2];
                ByteBuffer bb = ByteBuffer.wrap(rawBytes).order(ByteOrder.LITTLE_ENDIAN);
                for (int v : int32Values) bb.putShort((short) (v & 0xFFFF));
            } else {
                rawBytes = new byte[0];
            }
        } else {
            data = new float[0];
            rawBytes = new byte[0];
        }

        long[] dimsArr = dims.stream().mapToLong(Long::longValue).toArray();
        int elementCount = (data.length > 0) ? data.length : rawBytes.length;
        log.debug("Tensor '{}': dims={}, dataType={}, floats={}, rawBytes={}",
                name, Arrays.toString(dimsArr), dataType, data.length, rawBytes.length);
        return new OnnxTensor(name, dimsArr, dataType, data, rawBytes);
    }

    // ── ValueInfoProto (just extract name) ───────────────────────────────

    private static String parseValueInfoName(ByteBuffer buf, int end) {
        String name = null;
        while (buf.position() < end) {
            int loopStart = buf.position();
            int tag = readVarint32(buf);
            int fieldNum = tag >>> 3;
            int wireType = tag & 0x7;
            if (fieldNum == 1 && wireType == 2) {
                name = readString(buf);
            } else {
                skipField(buf, wireType);
            }
            if (buf.position() == loopStart) break;
        }
        return name;
    }

    // ── Protobuf primitives ──────────────────────────────────────────────

    private static int readVarint32(ByteBuffer buf) {
        int result = 0;
        int shift = 0;
        while (buf.hasRemaining()) {
            byte b = buf.get();
            result |= (b & 0x7F) << shift;
            if ((b & 0x80) == 0) return result;
            shift += 7;
        }
        return result;
    }

    private static long readVarint64(ByteBuffer buf) {
        long result = 0;
        int shift = 0;
        while (buf.hasRemaining()) {
            byte b = buf.get();
            result |= (long) (b & 0x7F) << shift;
            if ((b & 0x80) == 0) return result;
            shift += 7;
        }
        return result;
    }

    private static String readString(ByteBuffer buf) {
        int len = readVarint32(buf);
        byte[] bytes = new byte[len];
        buf.get(bytes);
        return new String(bytes, java.nio.charset.StandardCharsets.UTF_8);
    }

    private static void skipField(ByteBuffer buf, int wireType) {
        switch (wireType) {
            case 0 -> readVarint64(buf);          // varint
            case 1 -> buf.position(Math.min(buf.position() + 8, buf.limit()));  // 64-bit
            case 2 -> {                            // length-delimited
                int len = readVarint32(buf);
                // Clamp to limit so an inflated length (from external raw_data stubs
                // or a truncated file) cannot throw IllegalArgumentException.
                buf.position(Math.min(buf.position() + len, buf.limit()));
            }
            case 5 -> buf.position(Math.min(buf.position() + 4, buf.limit()));  // 32-bit
            default -> throw new RuntimeException("Unknown wire type: " + wireType);
        }
    }
}

