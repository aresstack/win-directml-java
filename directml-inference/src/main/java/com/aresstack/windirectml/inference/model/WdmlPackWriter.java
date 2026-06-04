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
import java.util.Map;
import java.util.Objects;

/**
 * Minimal writer for the internal Win-DirectML model package container.
 *
 * <p>v21 intentionally writes a <em>manifest-only</em> package. The payload is
 * not copied yet and the runtime still loads the already-proven ONNX import
 * path. This gives the next iteration a stable binary container with a header,
 * version marker and JSON manifest without risking the Qwen runtime.</p>
 *
 * <h2>Binary layout v1</h2>
 * <pre>{@code
 * 0x00  8 bytes   ASCII magic "WDMLPACK"
 * 0x08  int32le   container version (=1)
 * 0x0c  int32le   flags (bit 0 = manifest-only)
 * 0x10  int64le   manifest offset
 * 0x18  int64le   manifest length
 * 0x20  int64le   tensor directory offset (0 for v21 manifest-only)
 * 0x28  int64le   tensor directory length (0 for v21 manifest-only)
 * 0x30  int64le   payload offset (0 for v21 manifest-only)
 * 0x38  int64le   payload length (0 for v21 manifest-only)
 * 0x40  ...       UTF-8 JSON manifest
 * }</pre>
 */
public final class WdmlPackWriter {

    public static final String MAGIC = "WDMLPACK";
    public static final int VERSION = 1;
    public static final int HEADER_SIZE = 64;
    public static final int FLAG_MANIFEST_ONLY = 1;

    private static final ObjectMapper MAPPER = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);

    private WdmlPackWriter() {
    }

    /**
     * Writes a manifest-only {@code .wdmlpack} atomically.
     */
    public static Path writeManifestOnly(Path output, Map<String, Object> manifest) throws IOException {
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(manifest, "manifest");
        Path parent = output.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }

        byte[] json = MAPPER.writeValueAsBytes(manifest);
        ByteBuffer header = ByteBuffer.allocate(HEADER_SIZE).order(ByteOrder.LITTLE_ENDIAN);
        header.put(MAGIC.getBytes(StandardCharsets.US_ASCII));
        header.putInt(VERSION);
        header.putInt(FLAG_MANIFEST_ONLY);
        header.putLong(HEADER_SIZE);      // manifest offset
        header.putLong(json.length);      // manifest length
        header.putLong(0L);               // tensor directory offset (future)
        header.putLong(0L);               // tensor directory length (future)
        header.putLong(0L);               // payload offset (future)
        header.putLong(0L);               // payload length (future)

        Path tmp = output.resolveSibling(output.getFileName() + ".tmp");
        Files.write(tmp, header.array(), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        Files.write(tmp, json, StandardOpenOption.APPEND);
        try {
            Files.move(tmp, output, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException atomicMoveNotSupported) {
            Files.move(tmp, output, StandardCopyOption.REPLACE_EXISTING);
        }
        return output;
    }

    /**
     * Reads the JSON manifest from a v1 package. Intended for tests, tools and
     * the v22 runtime package front door. The package is memory-mapped so this
     * stays consistent with the runtime loading policy and avoids whole-file
     * {@code byte[]} reads.
     */
    public static Map<String, Object> readManifest(Path pack) throws IOException {
        Objects.requireNonNull(pack, "pack");
        try (FileChannel channel = FileChannel.open(pack, StandardOpenOption.READ)) {
            long fileSize = channel.size();
            if (fileSize < HEADER_SIZE) {
                throw new IOException("Invalid wdmlpack: file is smaller than header: " + pack);
            }
            if (fileSize > Integer.MAX_VALUE) {
                throw new IOException("Invalid wdmlpack: manifest-only package is unexpectedly large: " + pack
                        + " (" + fileSize + " bytes)");
            }
            MappedByteBuffer mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);
            mapped.order(ByteOrder.LITTLE_ENDIAN);
            ByteBuffer header = mapped.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
            header.limit(HEADER_SIZE);
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
            header.getInt(); // flags
            long manifestOffset = header.getLong();
            long manifestLength = header.getLong();
            if (manifestOffset < HEADER_SIZE || manifestLength < 0 || manifestOffset + manifestLength > fileSize) {
                throw new IOException("Invalid wdmlpack manifest range: offset=" + manifestOffset
                        + ", length=" + manifestLength + ", fileSize=" + fileSize);
            }
            ByteBuffer manifestBuffer = mapped.asReadOnlyBuffer();
            manifestBuffer.position(Math.toIntExact(manifestOffset));
            manifestBuffer.limit(Math.toIntExact(manifestOffset + manifestLength));
            byte[] json = new byte[Math.toIntExact(manifestLength)];
            manifestBuffer.get(json);
            return MAPPER.readValue(json, new TypeReference<>() {
            });
        }
    }
}
