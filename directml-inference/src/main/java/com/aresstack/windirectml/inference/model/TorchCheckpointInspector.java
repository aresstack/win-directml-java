package com.aresstack.windirectml.inference.model;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * Reads PyTorch ZIP state-dict checkpoints as data-only tensor metadata.
 *
 * <p>This class deliberately implements a tiny allowlist-based Pickle reader for
 * Tensor state_dict files. It never executes Python globals. Unsupported pickle
 * opcodes or reduce callables fail fast so the importer cannot become a general
 * PyTorch loader by accident.</p>
 */
public final class TorchCheckpointInspector {

    private TorchCheckpointInspector() {
    }

    public static TorchCheckpointInspection inspect(Path checkpoint) throws IOException {
        Objects.requireNonNull(checkpoint, "checkpoint");
        if (!Files.isRegularFile(checkpoint)) {
            throw new IOException("Torch checkpoint does not exist: " + checkpoint);
        }
        try (ZipFile zip = new ZipFile(checkpoint.toFile())) {
            ZipEntry dataPickle = findDataPickle(zip);
            String prefix = dataPickle.getName().substring(0, dataPickle.getName().length() - "data.pkl".length());
            byte[] pickleBytes = readAll(zip, dataPickle);
            Map<String, TensorRef> tensors = new RestrictedPickleReader(pickleBytes).readStateDict();
            List<TorchCheckpointTensor> inspected = new ArrayList<>();
            long declaredTensorBytes = 0L;
            long storageBytes = 0L;
            for (Map.Entry<String, TensorRef> entry : tensors.entrySet()) {
                TensorRef tensor = entry.getValue();
                String storageEntryName = prefix + "data/" + tensor.storage().key();
                ZipEntry storageEntry = zip.getEntry(storageEntryName);
                long storageByteLength = storageEntry == null ? 0L : storageEntry.getSize();
                if (storageByteLength < 0) {
                    storageByteLength = 0L;
                }
                long tensorByteLength = tensor.tensorByteLength();
                declaredTensorBytes = Math.addExact(declaredTensorBytes, tensorByteLength);
                storageBytes = Math.addExact(storageBytes, storageByteLength);
                inspected.add(new TorchCheckpointTensor(entry.getKey(), tensor.storage().dataType(), tensor.size(),
                        tensor.stride(), tensor.storage().key(), storageEntryName, tensor.storageOffset(),
                        tensor.storage().elements(), storageByteLength, tensorByteLength, storageEntry != null));
            }
            inspected.sort(Comparator.comparing(TorchCheckpointTensor::name));
            return new TorchCheckpointInspection(checkpoint, prefix, inspected.size(), declaredTensorBytes,
                    storageBytes, inspected);
        } catch (java.util.zip.ZipException e) {
            throw new IOException("Unsupported PyTorch checkpoint format: expected torch.save ZIP archive", e);
        }
    }

    private static ZipEntry findDataPickle(ZipFile zip) throws IOException {
        List<? extends ZipEntry> entries = zip.stream()
                .filter(entry -> !entry.isDirectory() && entry.getName().endsWith("data.pkl"))
                .toList();
        if (entries.isEmpty()) {
            throw new IOException("Unsupported PyTorch checkpoint: missing data.pkl");
        }
        if (entries.size() > 1) {
            throw new IOException("Unsupported PyTorch checkpoint: multiple data.pkl entries");
        }
        return entries.get(0);
    }

    private static byte[] readAll(ZipFile zip, ZipEntry entry) throws IOException {
        try (InputStream in = zip.getInputStream(entry); ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            in.transferTo(out);
            return out.toByteArray();
        }
    }

    private record PyGlobal(String module, String name) {
        String qualifiedName() {
            return module + "." + name;
        }
    }

    private record StorageRef(SourceTensorDataType dataType, String key, String location, long elements) {
    }

    private record TensorRef(StorageRef storage, long storageOffset, long[] size, long[] stride) {
        TensorRef {
            size = size == null ? new long[0] : size.clone();
            stride = stride == null ? new long[0] : stride.clone();
        }

        @Override
        public long[] size() {
            return size.clone();
        }

        @Override
        public long[] stride() {
            return stride.clone();
        }

        long tensorByteLength() {
            long elements = 1L;
            for (long dim : size) {
                elements = Math.multiplyExact(elements, dim);
            }
            return Math.multiplyExact(elements, storage.dataType().bytesPerElement());
        }
    }

    private static final class Mark {
        private static final Mark INSTANCE = new Mark();
    }

    private static final class None {
        private static final None INSTANCE = new None();

        @Override
        public String toString() {
            return "None";
        }
    }

    private static final class RestrictedPickleReader {
        private final DataInputStream in;
        private final ArrayDeque<Object> stack = new ArrayDeque<>();
        private final Map<Integer, Object> memo = new LinkedHashMap<>();
        private int nextMemoIndex;

        RestrictedPickleReader(byte[] pickleBytes) {
            this.in = new DataInputStream(new ByteArrayInputStream(pickleBytes));
        }

        Map<String, TensorRef> readStateDict() throws IOException {
            while (true) {
                int opcode = in.read();
                if (opcode < 0) {
                    throw new IOException("Unexpected end of pickle stream before STOP");
                }
                if (opcode == '.') {
                    Object result = pop();
                    if (!(result instanceof Map<?, ?> map)) {
                        throw new IOException("Unsupported PyTorch checkpoint: pickle root is not a state_dict");
                    }
                    return convertStateDict(map);
                }
                handle(opcode);
            }
        }

        private Map<String, TensorRef> convertStateDict(Map<?, ?> map) throws IOException {
            LinkedHashMap<String, TensorRef> tensors = new LinkedHashMap<>();
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                if (!(entry.getKey() instanceof String name)) {
                    throw new IOException("Unsupported state_dict key type: " + entry.getKey());
                }
                if (!(entry.getValue() instanceof TensorRef tensor)) {
                    throw new IOException("Unsupported state_dict value for " + name + ": " + entry.getValue());
                }
                tensors.put(name, tensor);
            }
            return tensors;
        }

        private void handle(int opcode) throws IOException {
            switch (opcode) {
                case 0x80 -> readProto();
                case 0x95 -> skipFrame();
                case '(' -> stack.push(Mark.INSTANCE);
                case '}' -> stack.push(new LinkedHashMap<>());
                case ']' -> stack.push(new ArrayList<>());
                case ')' -> stack.push(List.of());
                case 'N' -> stack.push(None.INSTANCE);
                case 0x88 -> stack.push(Boolean.TRUE);
                case 0x89 -> stack.push(Boolean.FALSE);
                case 'J' -> stack.push((long) readIntLE());
                case 'K' -> stack.push((long) readU8());
                case 'M' -> stack.push((long) readU16LE());
                case 0x8a -> stack.push(readLong1());
                case 0x8b -> stack.push(readLong4());
                case 'X' -> stack.push(readUtf8(readIntLE()));
                case 0x8c -> stack.push(readUtf8(readU8()));
                case 0x8d -> stack.push(readUtf8(readLongAsInt(readLongLE())));
                case 'U' -> stack.push(readUtf8(readU8()));
                case 'c' -> stack.push(readGlobal());
                case 0x93 -> stack.push(readStackGlobal());
                case 't' -> stack.push(popToMark());
                case 0x85 -> stack.push(tuple1());
                case 0x86 -> stack.push(tuple2());
                case 0x87 -> stack.push(tuple3());
                case 'e' -> appendItems();
                case 's' -> setItem();
                case 'u' -> setItems();
                case 'R' -> stack.push(reduce());
                case 'Q' -> stack.push(persistentLoad(pop()));
                case 'q' -> memo.put(readU8(), peek());
                case 'r' -> memo.put(readIntLE(), peek());
                case 'p' -> memo.put(Integer.parseInt(readLine()), peek());
                case 0x94 -> memo.put(nextMemoIndex++, peek());
                case 'h' -> stack.push(memoGet(readU8()));
                case 'j' -> stack.push(memoGet(readIntLE()));
                case 'g' -> stack.push(memoGet(Integer.parseInt(readLine())));
                case 'b' -> ignoreBuildState();
                default -> throw new IOException("Unsupported pickle opcode 0x" + Integer.toHexString(opcode)
                        + " while inspecting Torch checkpoint");
            }
        }

        private void readProto() throws IOException {
            int protocol = readU8();
            if (protocol < 2 || protocol > 5) {
                throw new IOException("Unsupported pickle protocol for Torch checkpoint: " + protocol);
            }
        }

        private void skipFrame() throws IOException {
            long frameSize = readLongLE();
            if (frameSize < 0) {
                throw new IOException("Invalid pickle frame size: " + frameSize);
            }
        }

        private PyGlobal readGlobal() throws IOException {
            String module = readLine();
            String name = readLine();
            return new PyGlobal(module, name);
        }

        private PyGlobal readStackGlobal() throws IOException {
            Object name = pop();
            Object module = pop();
            if (!(module instanceof String moduleName) || !(name instanceof String objectName)) {
                throw new IOException("STACK_GLOBAL requires module/name strings");
            }
            return new PyGlobal(moduleName, objectName);
        }

        private Object reduce() throws IOException {
            Object args = pop();
            Object callable = pop();
            if (!(callable instanceof PyGlobal global)) {
                throw new IOException("Unsupported REDUCE callable: " + callable);
            }
            List<?> tupleArgs = requireTuple(args, "REDUCE args for " + global.qualifiedName());
            return switch (global.qualifiedName()) {
                case "collections.OrderedDict" -> new LinkedHashMap<>();
                case "torch._utils._rebuild_tensor_v2", "torch._utils._rebuild_tensor" -> rebuildTensor(global, tupleArgs);
                default -> throw new IOException("Unsupported REDUCE callable in Torch checkpoint: "
                        + global.qualifiedName());
            };
        }

        private TensorRef rebuildTensor(PyGlobal global, List<?> args) throws IOException {
            if (args.size() < 4) {
                throw new IOException("Invalid " + global.qualifiedName() + " arguments: " + args.size());
            }
            if (!(args.get(0) instanceof StorageRef storage)) {
                throw new IOException("Tensor rebuild does not reference a supported storage");
            }
            long storageOffset = asLong(args.get(1), "storage offset");
            long[] size = asLongArray(requireTuple(args.get(2), "tensor size"), "tensor size");
            long[] stride = asLongArray(requireTuple(args.get(3), "tensor stride"), "tensor stride");
            return new TensorRef(storage, storageOffset, size, stride);
        }

        private StorageRef persistentLoad(Object persistentId) throws IOException {
            List<?> tuple = requireTuple(persistentId, "persistent storage id");
            if (tuple.size() < 5 || !"storage".equals(tuple.get(0))) {
                throw new IOException("Unsupported persistent id in Torch checkpoint: " + tuple);
            }
            SourceTensorDataType dataType = storageDataType(tuple.get(1));
            String key = String.valueOf(tuple.get(2));
            String location = String.valueOf(tuple.get(3));
            long elements = asLong(tuple.get(4), "storage elements");
            return new StorageRef(dataType, key, location, elements);
        }

        private SourceTensorDataType storageDataType(Object typeObject) throws IOException {
            String name;
            if (typeObject instanceof PyGlobal global) {
                name = global.name();
            } else if (typeObject instanceof String text) {
                name = text;
            } else {
                throw new IOException("Unsupported storage type object: " + typeObject);
            }
            return switch (name) {
                case "FloatStorage" -> SourceTensorDataType.FLOAT;
                case "HalfStorage" -> SourceTensorDataType.FLOAT16;
                case "BFloat16Storage" -> SourceTensorDataType.BFLOAT16;
                case "DoubleStorage" -> SourceTensorDataType.DOUBLE;
                case "LongStorage" -> SourceTensorDataType.INT64;
                case "IntStorage" -> SourceTensorDataType.INT32;
                case "ShortStorage" -> SourceTensorDataType.INT16;
                case "CharStorage" -> SourceTensorDataType.INT8;
                case "ByteStorage" -> SourceTensorDataType.UINT8;
                case "BoolStorage" -> SourceTensorDataType.BOOL;
                default -> throw new IOException("Unsupported Torch storage dtype: " + name);
            };
        }

        private List<Object> popToMark() throws IOException {
            ArrayList<Object> values = new ArrayList<>();
            while (!stack.isEmpty()) {
                Object value = pop();
                if (value == Mark.INSTANCE) {
                    java.util.Collections.reverse(values);
                    return values;
                }
                values.add(value);
            }
            throw new IOException("Pickle MARK not found");
        }

        private List<Object> tuple1() throws IOException {
            return List.of(pop());
        }

        private List<Object> tuple2() throws IOException {
            Object second = pop();
            Object first = pop();
            return List.of(first, second);
        }

        private List<Object> tuple3() throws IOException {
            Object third = pop();
            Object second = pop();
            Object first = pop();
            return List.of(first, second, third);
        }

        @SuppressWarnings("unchecked")
        private void appendItems() throws IOException {
            List<Object> values = popToMark();
            Object target = peek();
            if (!(target instanceof List<?> list)) {
                throw new IOException("APPENDS target is not a list");
            }
            ((List<Object>) list).addAll(values);
        }

        @SuppressWarnings("unchecked")
        private void setItem() throws IOException {
            Object value = pop();
            Object key = pop();
            Object target = peek();
            if (!(target instanceof Map<?, ?> map)) {
                throw new IOException("SETITEM target is not a dict");
            }
            ((Map<Object, Object>) map).put(key, value);
        }

        @SuppressWarnings("unchecked")
        private void setItems() throws IOException {
            List<Object> values = popToMark();
            Object target = peek();
            if (!(target instanceof Map<?, ?> map)) {
                throw new IOException("SETITEMS target is not a dict");
            }
            if (values.size() % 2 != 0) {
                throw new IOException("SETITEMS requires key/value pairs");
            }
            Map<Object, Object> typed = (Map<Object, Object>) map;
            for (int i = 0; i < values.size(); i += 2) {
                typed.put(values.get(i), values.get(i + 1));
            }
        }

        private void ignoreBuildState() throws IOException {
            Object state = pop();
            Object instance = peek();
            if (instance instanceof Map<?, ?>) {
                return;
            }
            throw new IOException("Unsupported BUILD target in Torch checkpoint: " + instance + " state=" + state);
        }

        private Object memoGet(int index) throws IOException {
            if (!memo.containsKey(index)) {
                throw new IOException("Invalid pickle memo reference: " + index);
            }
            return memo.get(index);
        }

        private Object pop() throws IOException {
            if (stack.isEmpty()) {
                throw new IOException("Pickle stack underflow");
            }
            return stack.pop();
        }

        private Object peek() throws IOException {
            if (stack.isEmpty()) {
                throw new IOException("Pickle stack underflow");
            }
            return stack.peek();
        }

        private List<?> requireTuple(Object value, String context) throws IOException {
            if (!(value instanceof List<?> list)) {
                throw new IOException(context + " must be a tuple/list, got " + value);
            }
            return list;
        }

        private long[] asLongArray(List<?> values, String context) throws IOException {
            long[] out = new long[values.size()];
            for (int i = 0; i < values.size(); i++) {
                out[i] = asLong(values.get(i), context + "[" + i + "]");
            }
            return out;
        }

        private long asLong(Object value, String context) throws IOException {
            if (value instanceof Number number) {
                return number.longValue();
            }
            throw new IOException(context + " must be integer, got " + value);
        }

        private int readU8() throws IOException {
            int value = in.read();
            if (value < 0) {
                throw new EOFException();
            }
            return value;
        }

        private int readU16LE() throws IOException {
            int b0 = readU8();
            int b1 = readU8();
            return b0 | (b1 << 8);
        }

        private int readIntLE() throws IOException {
            int b0 = readU8();
            int b1 = readU8();
            int b2 = readU8();
            int b3 = readU8();
            return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }

        private long readLongLE() throws IOException {
            long value = 0L;
            for (int i = 0; i < Long.BYTES; i++) {
                value |= ((long) readU8()) << (8 * i);
            }
            return value;
        }

        private long readLong1() throws IOException {
            int length = readU8();
            return readLittleEndianInteger(length);
        }

        private long readLong4() throws IOException {
            int length = readIntLE();
            if (length < 0 || length > 16) {
                throw new IOException("Unsupported LONG4 byte length: " + length);
            }
            return readLittleEndianInteger(length);
        }

        private long readLittleEndianInteger(int length) throws IOException {
            if (length == 0) {
                return 0L;
            }
            byte[] bytes = new byte[length];
            in.readFully(bytes);
            byte[] bigEndian = new byte[length];
            for (int i = 0; i < length; i++) {
                bigEndian[i] = bytes[length - 1 - i];
            }
            return new BigInteger(1, bigEndian).longValueExact();
        }

        private String readUtf8(int length) throws IOException {
            if (length < 0) {
                throw new IOException("Invalid string length: " + length);
            }
            byte[] bytes = new byte[length];
            in.readFully(bytes);
            return new String(bytes, StandardCharsets.UTF_8);
        }

        private String readLine() throws IOException {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            while (true) {
                int value = in.read();
                if (value < 0) {
                    throw new EOFException();
                }
                if (value == '\n') {
                    return out.toString(StandardCharsets.UTF_8);
                }
                out.write(value);
            }
        }

        private int readLongAsInt(long value) throws IOException {
            if (value < 0 || value > Integer.MAX_VALUE) {
                throw new IOException("String length exceeds Java array limit: " + value);
            }
            return (int) value;
        }
    }
}
