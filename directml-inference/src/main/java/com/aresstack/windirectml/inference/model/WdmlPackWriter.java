package com.aresstack.windirectml.inference.model;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Writer/reader helpers for the internal Win-DirectML model package container.
 *
 * <h2>Binary layout v1</h2>
 * <pre>{@code
 * 0x00  8 bytes   ASCII magic "WDMLPACK"
 * 0x08  int32le   container version (=1)
 * 0x0c  int32le   flags (bit 0 = manifest-only)
 * 0x10  int64le   manifest offset
 * 0x18  int64le   manifest length
 * 0x20  int64le   tensor directory offset (0; currently inside JSON manifest)
 * 0x28  int64le   tensor directory length (0; currently inside JSON manifest)
 * 0x30  int64le   payload offset (0 for manifest-only)
 * 0x38  int64le   payload length (0 for manifest-only)
 * 0x40  ...       UTF-8 JSON manifest
 * ...   padding   zero padding to PAYLOAD_ALIGNMENT
 * ...   payload   concatenated tensor payloads; tensor offsets are relative to payload offset
 * }</pre>
 */
public final class WdmlPackWriter {

    public static final String MAGIC = "WDMLPACK";
    public static final int VERSION = 1;
    public static final int HEADER_SIZE = 64;
    public static final int FLAG_MANIFEST_ONLY = 1;
    public static final int PAYLOAD_ALIGNMENT = 4096;

    private static final ObjectMapper MAPPER = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);

    private static final byte[] ZEROES = new byte[8192];

    private WdmlPackWriter() {
    }

    public record Header(int version, int flags,
                         long manifestOffset, long manifestLength,
                         long tensorDirectoryOffset, long tensorDirectoryLength,
                         long payloadOffset, long payloadLength) {
        public boolean manifestOnly() {
            return (flags & FLAG_MANIFEST_ONLY) != 0;
        }

        public boolean payloadIncluded() {
            return payloadOffset > 0 && payloadLength > 0 && !manifestOnly();
        }
    }

    @FunctionalInterface
    public interface PayloadWriter {
        void writeTo(FileChannel channel) throws IOException;
    }

    public record PayloadEntry(String name, long relativeOffset, long byteLength, PayloadWriter writer) {
        public PayloadEntry {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(writer, "writer");
            if (relativeOffset < 0) {
                throw new IllegalArgumentException("relativeOffset must be >= 0");
            }
            if (byteLength < 0) {
                throw new IllegalArgumentException("byteLength must be >= 0");
            }
        }
    }

    /**
     * Writes a manifest-only {@code .wdmlpack} atomically.
     */
    public static Path writeManifestOnly(Path output, Map<String, Object> manifest) throws IOException {
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(manifest, "manifest");
        byte[] json = MAPPER.writeValueAsBytes(manifest);
        return writeContainer(output, json, FLAG_MANIFEST_ONLY, 0L, 0L, List.of());
    }

    /**
     * Writes a payload-carrying {@code .wdmlpack} atomically. Tensor payload offsets
     * stored in the JSON manifest are relative to the container payload base.
     */
    public static Path writeWithPayload(Path output, Map<String, Object> manifest,
                                        List<PayloadEntry> payloadEntries,
                                        long payloadLength) throws IOException {
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(manifest, "manifest");
        Objects.requireNonNull(payloadEntries, "payloadEntries");
        if (payloadLength < 0) {
            throw new IllegalArgumentException("payloadLength must be >= 0");
        }
        byte[] json = MAPPER.writeValueAsBytes(manifest);
        long payloadOffset = align(HEADER_SIZE + json.length, PAYLOAD_ALIGNMENT);
        return writeContainer(output, json, 0, payloadOffset, payloadLength, payloadEntries);
    }

    public static Map<String, Object> readManifest(Path pack) throws IOException {
        Header header = readHeader(pack);
        long fileSize = Files.size(pack);
        if (fileSize > Integer.MAX_VALUE) {
            throw new IOException("wdmlpack is too large for the current lightweight mmap reader: " + pack
                    + " (" + fileSize + " bytes)");
        }
        try (FileChannel channel = FileChannel.open(pack, StandardOpenOption.READ)) {
            MappedByteBuffer mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);
            mapped.order(ByteOrder.LITTLE_ENDIAN);
            return readManifest(mapped, header, fileSize);
        }
    }

    public static Header readHeader(Path pack) throws IOException {
        Objects.requireNonNull(pack, "pack");
        try (FileChannel channel = FileChannel.open(pack, StandardOpenOption.READ)) {
            long fileSize = channel.size();
            if (fileSize < HEADER_SIZE) {
                throw new IOException("Invalid wdmlpack: file is smaller than header: " + pack);
            }
            ByteBuffer header = ByteBuffer.allocate(HEADER_SIZE).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(header, 0);
            header.flip();
            byte[] magic = new byte[8];
            header.get(magic);
            String magicText = new String(magic, StandardCharsets.US_ASCII);
            if (!MAGIC.equals(magicText)) {
                throw new IOException("Invalid wdmlpack magic: " + magicText);
            }
            int version = header.getInt();
            if (version != VERSION) {
                throw new IOException("Unsupported wdmlpack version: " + version);
            }
            int flags = header.getInt();
            long manifestOffset = header.getLong();
            long manifestLength = header.getLong();
            long tensorDirectoryOffset = header.getLong();
            long tensorDirectoryLength = header.getLong();
            long payloadOffset = header.getLong();
            long payloadLength = header.getLong();
            validateRange("manifest", manifestOffset, manifestLength, fileSize, HEADER_SIZE);
            if (payloadOffset != 0 || payloadLength != 0) {
                validateRange("payload", payloadOffset, payloadLength, fileSize, HEADER_SIZE);
            }
            return new Header(version, flags, manifestOffset, manifestLength,
                    tensorDirectoryOffset, tensorDirectoryLength, payloadOffset, payloadLength);
        }
    }

    public static long align(long value, int alignment) {
        long mask = alignment - 1L;
        return (value + mask) & ~mask;
    }

    private static Path writeContainer(Path output, byte[] json, int flags,
                                       long payloadOffset, long payloadLength,
                                       List<PayloadEntry> payloadEntries) throws IOException {
        Path parent = output.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path tmp = output.resolveSibling(output.getFileName() + ".tmp");
        Files.deleteIfExists(tmp);
        try (FileChannel channel = FileChannel.open(tmp,
                StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE, StandardOpenOption.READ)) {
            ByteBuffer header = ByteBuffer.allocate(HEADER_SIZE).order(ByteOrder.LITTLE_ENDIAN);
            header.put(MAGIC.getBytes(StandardCharsets.US_ASCII));
            header.putInt(VERSION);
            header.putInt(flags);
            header.putLong(HEADER_SIZE);
            header.putLong(json.length);
            header.putLong(0L); // tensor directory offset; JSON manifest carries it for now
            header.putLong(0L); // tensor directory length
            header.putLong(payloadOffset);
            header.putLong(payloadLength);
            header.flip();
            channel.write(header);
            channel.write(ByteBuffer.wrap(json));

            if (payloadLength > 0) {
                writeZeroPadding(channel, payloadOffset - channel.position());
                for (PayloadEntry entry : payloadEntries) {
                    long expectedStart = payloadOffset + entry.relativeOffset();
                    writeZeroPadding(channel, expectedStart - channel.position());
                    long before = channel.position();
                    entry.writer().writeTo(channel);
                    long written = channel.position() - before;
                    if (written != entry.byteLength()) {
                        throw new IOException("Payload writer for '" + entry.name() + "' wrote " + written
                                + " bytes, expected " + entry.byteLength());
                    }
                }
                writeZeroPadding(channel, payloadOffset + payloadLength - channel.position());
            }
        }
        try {
            Files.move(tmp, output, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException atomicMoveNotSupported) {
            Files.move(tmp, output, StandardCopyOption.REPLACE_EXISTING);
        }
        return output;
    }

    private static Map<String, Object> readManifest(MappedByteBuffer mapped, Header header, long fileSize) throws IOException {
        if (header.manifestOffset() < HEADER_SIZE || header.manifestLength() < 0
                || header.manifestOffset() + header.manifestLength() > fileSize) {
            throw new IOException("Invalid wdmlpack manifest range: offset=" + header.manifestOffset()
                    + ", length=" + header.manifestLength() + ", fileSize=" + fileSize);
        }
        ByteBuffer manifestBuffer = mapped.asReadOnlyBuffer();
        manifestBuffer.position(Math.toIntExact(header.manifestOffset()));
        manifestBuffer.limit(Math.toIntExact(header.manifestOffset() + header.manifestLength()));
        byte[] json = new byte[Math.toIntExact(header.manifestLength())];
        manifestBuffer.get(json);
        return MAPPER.readValue(json, new TypeReference<>() {
        });
    }

    private static void validateRange(String label, long offset, long length, long fileSize, long minOffset) throws IOException {
        if (offset < minOffset || length < 0 || offset + length < offset || offset + length > fileSize) {
            throw new IOException("Invalid wdmlpack " + label + " range: offset=" + offset
                    + ", length=" + length + ", fileSize=" + fileSize);
        }
    }

    private static void writeZeroPadding(FileChannel channel, long count) throws IOException {
        if (count < 0) {
            throw new IOException("wdmlpack writer attempted to move backwards by " + (-count) + " bytes");
        }
        while (count > 0) {
            int chunk = (int) Math.min(count, ZEROES.length);
            channel.write(ByteBuffer.wrap(ZEROES, 0, chunk));
            count -= chunk;
        }
    }
}
