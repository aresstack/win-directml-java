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

class TorchCheckpointInspectorTest {

    @TempDir
    Path tempDir;

    @Test
    void inspectsRestrictedTorchStateDictCheckpoint() throws Exception {
        Path checkpoint = tempDir.resolve("pytorch_model.bin");
        writeZip(checkpoint, torchStateDictPickle(), new byte[24]);

        TorchCheckpointInspection inspection = TorchCheckpointInspector.inspect(checkpoint);

        assertEquals("archive/", inspection.archivePrefix());
        assertEquals(1, inspection.tensorCount());
        assertEquals(24, inspection.declaredTensorBytes());
        assertEquals(24, inspection.storageBytes());
        TorchCheckpointTensor tensor = inspection.tensors().get(0);
        assertEquals("encoder.block.0.layer.0.SelfAttention.q.weight", tensor.name());
        assertEquals(SourceTensorDataType.FLOAT, tensor.dataType());
        assertArrayEquals(new long[]{2, 3}, tensor.shape());
        assertArrayEquals(new long[]{3, 1}, tensor.stride());
        assertEquals("0", tensor.storageKey());
        assertEquals("archive/data/0", tensor.storageEntryName());
        assertTrue(tensor.storageEntryPresent());
    }

    @Test
    void rejectsUnsafeReduceCallable() throws Exception {
        Path checkpoint = tempDir.resolve("unsafe.bin");
        writeZip(checkpoint, unsafePickle(), new byte[0]);

        java.io.IOException thrown = assertThrows(java.io.IOException.class,
                () -> TorchCheckpointInspector.inspect(checkpoint));

        assertTrue(thrown.getMessage().contains("Unsupported REDUCE callable"));
    }

    @Test
    void rejectsNonZipLegacyCheckpoint() {
        Path checkpoint = tempDir.resolve("legacy.bin");
        assertDoesNotThrow(() -> java.nio.file.Files.writeString(checkpoint, "not a zip"));

        java.io.IOException thrown = assertThrows(java.io.IOException.class,
                () -> TorchCheckpointInspector.inspect(checkpoint));

        assertTrue(thrown.getMessage().contains("expected torch.save ZIP archive"));
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

    private static byte[] torchStateDictPickle() throws Exception {
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
        out.binInt1(2);
        out.binInt1(3);
        out.tuple();
        out.mark();
        out.binInt1(3);
        out.binInt1(1);
        out.tuple();
        out.newFalse();
        out.emptyDict();
        out.tuple();
        out.reduce();
        out.setItems();
        out.stop();
        return out.toByteArray();
    }

    private static byte[] unsafePickle() throws Exception {
        PickleWriter out = new PickleWriter();
        out.proto(2);
        out.global("os", "system");
        out.emptyTuple();
        out.reduce();
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

        void emptyTuple() {
            out.write(')');
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
