package com.aresstack.windirectml.inference.model;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import static org.junit.jupiter.api.Assertions.*;

class TorchCheckpointModelSourceTest {

    @TempDir
    Path tempDir;

    @Test
    void loadsRestrictedCheckpointAsSourceTensorCatalog() throws Exception {
        Path checkpoint = tempDir.resolve("pytorch_model.bin");
        byte[] storage = ByteBuffer.allocate(24)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putFloat(1.0f)
                .putFloat(2.0f)
                .putFloat(3.0f)
                .putFloat(4.0f)
                .putFloat(5.0f)
                .putFloat(6.0f)
                .array();
        writeZip(checkpoint, torchStateDictPickle(new long[]{2, 3}, new long[]{3, 1}), storage);

        SourceTensorCatalog catalog = TorchCheckpointModelSource.of(checkpoint).load();

        assertEquals(1, catalog.size());
        SourceTensor tensor = catalog.get("encoder.block.0.layer.0.SelfAttention.q.weight");
        assertNotNull(tensor);
        assertEquals(SourceTensorDataType.FLOAT, tensor.dataType());
        assertArrayEquals(new long[]{2, 3}, tensor.dims());
        assertEquals(24, tensor.byteLength());
        ByteBuffer data = tensor.payloadBuffer().order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(1.0f, data.getFloat(0), 0.0f);
        assertEquals(6.0f, data.getFloat(20), 0.0f);
    }

    @Test
    void rejectsNonCompactTensorStride() throws Exception {
        Path checkpoint = tempDir.resolve("noncompact.bin");
        writeZip(checkpoint, torchStateDictPickle(new long[]{2, 3}, new long[]{1, 2}), new byte[24]);

        java.io.IOException thrown = assertThrows(java.io.IOException.class,
                () -> TorchCheckpointModelSource.of(checkpoint).load());

        assertTrue(thrown.getMessage().contains("not compact row-major"));
    }

    private static void writeZip(Path checkpoint, byte[] dataPickle, byte[] storage) throws Exception {
        try (ZipOutputStream zip = new ZipOutputStream(java.nio.file.Files.newOutputStream(checkpoint))) {
            zip.putNextEntry(new ZipEntry("archive/data.pkl"));
            zip.write(dataPickle);
            zip.closeEntry();
            zip.putNextEntry(new ZipEntry("archive/data/0"));
            zip.write(storage);
            zip.closeEntry();
            zip.putNextEntry(new ZipEntry("archive/version"));
            zip.write("3\n".getBytes(StandardCharsets.UTF_8));
            zip.closeEntry();
        }
    }

    private static byte[] torchStateDictPickle(long[] shape, long[] stride) throws Exception {
        PickleWriter out = new PickleWriter();
        out.proto(2);
        out.emptyDict();
        out.mark();
        out.binUnicode("encoder.block.0.layer.0.SelfAttention.q.weight");
        out.global("torch._utils", "_rebuild_tensor_v2");
        out.mark();
        out.mark();
        out.binUnicode("storage");
        out.global("torch", "FloatStorage");
        out.binUnicode("0");
        out.binUnicode("cpu");
        out.binInt1(6);
        out.tuple();
        out.binPersId();
        out.binInt1(0);
        out.mark();
        for (long value : shape) {
            out.binInt1((int) value);
        }
        out.tuple();
        out.mark();
        for (long value : stride) {
            out.binInt1((int) value);
        }
        out.tuple();
        out.newFalse();
        out.emptyDict();
        out.tuple();
        out.reduce();
        out.setItems();
        out.stop();
        return out.toByteArray();
    }

    private static final class PickleWriter {
        private final ByteArrayOutputStream out = new ByteArrayOutputStream();

        void proto(int version) {
            out.write(0x80);
            out.write(version);
        }

        void mark() {
            out.write('(');
        }

        void emptyDict() {
            out.write('}');
        }

        void tuple() {
            out.write('t');
        }

        void reduce() {
            out.write('R');
        }

        void setItems() {
            out.write('u');
        }

        void binPersId() {
            out.write('Q');
        }

        void newFalse() {
            out.write(0x89);
        }

        void stop() {
            out.write('.');
        }

        void global(String module, String name) {
            out.write('c');
            writeAsciiLine(module);
            writeAsciiLine(name);
        }

        void binInt1(int value) {
            out.write('K');
            out.write(value);
        }

        void binUnicode(String value) {
            byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
            out.write('X');
            out.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(bytes.length).array(), 0, 4);
            out.write(bytes, 0, bytes.length);
        }

        byte[] toByteArray() {
            return out.toByteArray();
        }

        private void writeAsciiLine(String value) {
            byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
            out.write(bytes, 0, bytes.length);
            out.write('\n');
        }
    }
}
