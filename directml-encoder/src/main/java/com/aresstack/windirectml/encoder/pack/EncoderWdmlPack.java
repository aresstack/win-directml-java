package com.aresstack.windirectml.encoder.pack;

import com.aresstack.windirectml.encoder.safetensors.SafetensorsException;
import com.aresstack.windirectml.encoder.safetensors.SafetensorsReader;
import com.aresstack.windirectml.inference.model.RuntimeModelPackage;
import com.aresstack.windirectml.inference.model.SourceFingerprint;
import com.aresstack.windirectml.inference.model.WdmlPackWriter;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * wdmlpack compiler + weight reader for BERT-style encoders (MiniLM/E5) and cross-encoder rerankers.
 *
 * <p>To guarantee bit-exact parity with the previous direct-SafeTensors path, the package stores the
 * model's {@code model.safetensors} <b>image verbatim</b> as its payload. The runtime reads the weights
 * back from the package through {@link SafetensorsReader#openFromBuffer}, i.e. the exact same parser and
 * weight builders as before - only the <em>source</em> changes from a loose file to the package. This
 * reuses the shared {@link WdmlPackWriter}/{@link RuntimeModelPackage} container (no second parser, no
 * copied reader) and is intentionally minimal; a native per-tensor encoder layout can come later.</p>
 */
public final class EncoderWdmlPack {

    /** Package file name for embedding encoders (MiniLM/E5). */
    public static final String ENCODER_PACKAGE_FILE = "encoder.wdmlpack";
    /** Package file name for cross-encoder rerankers. */
    public static final String RERANKER_PACKAGE_FILE = "reranker.wdmlpack";

    public static final String FAMILY_ENCODER = "bert-encoder";
    public static final String FAMILY_RERANKER = "bert-reranker";

    public static final String RUNTIME_LOAD_MODE = "encoder-safetensors-payload";
    private static final String SAFETENSORS_SOURCE = "model.safetensors";

    private EncoderWdmlPack() {
    }

    /** Compile {@code modelDir/model.safetensors} into a payload-backed wdmlpack. The only write path. */
    public static Path compile(Path modelDir, Path output, String modelFamily) throws IOException {
        Path dir = modelDir.toAbsolutePath().normalize();
        Path safetensors = dir.resolve(SAFETENSORS_SOURCE);
        if (!Files.isRegularFile(safetensors)) {
            throw new IOException("Missing " + SAFETENSORS_SOURCE + " to convert in " + dir);
        }
        byte[] image = Files.readAllBytes(safetensors);
        Path out = output.toAbsolutePath().normalize();
        Map<String, Object> manifest = buildManifest(modelFamily, safetensors, image.length);
        List<WdmlPackWriter.PayloadEntry> entries = List.of(new WdmlPackWriter.PayloadEntry(
                SAFETENSORS_SOURCE, 0L, image.length, channel -> {
            ByteBuffer buffer = ByteBuffer.wrap(image);
            while (buffer.hasRemaining()) {
                channel.write(buffer);
            }
        }));
        WdmlPackWriter.writeWithPayload(out, manifest, entries, image.length);
        return out;
    }

    /**
     * Open a {@link SafetensorsReader} over the SafeTensors image stored in the package payload. The
     * existing weight builders consume this exactly like a loose {@code model.safetensors}.
     */
    public static SafetensorsReader openWeightsReader(Path packagePath) throws IOException, SafetensorsException {
        RuntimeModelPackage pkg = RuntimeModelPackage.open(packagePath);
        WdmlPackWriter.Header header = pkg.header();
        if (!header.payloadIncluded()) {
            throw new IOException("Encoder wdmlpack has no weight payload: " + packagePath);
        }
        // Read the SafeTensors image into a heap buffer (no lingering memory-map / file lock on the
        // package). The downstream weight builders copy per-tensor anyway, matching the prior profile.
        byte[] image = new byte[Math.toIntExact(header.payloadLength())];
        try (FileChannel channel = FileChannel.open(packagePath, StandardOpenOption.READ)) {
            ByteBuffer buffer = ByteBuffer.wrap(image);
            long position = header.payloadOffset();
            while (buffer.hasRemaining()) {
                int read = channel.read(buffer, position + buffer.position());
                if (read < 0) {
                    throw new IOException("Truncated encoder wdmlpack payload: " + packagePath);
                }
            }
        }
        return SafetensorsReader.openFromBuffer(ByteBuffer.wrap(image).order(ByteOrder.LITTLE_ENDIAN));
    }

    private static Map<String, Object> buildManifest(String modelFamily, Path source, long payloadBytes)
            throws IOException {
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("format", "wdmlpack");
        root.put("version", WdmlPackWriter.VERSION);
        root.put("createdAt", Instant.now().toString());
        root.put("modelFamily", modelFamily);
        root.put("architecture", "bert");
        root.put("sourceFormat", "safetensors");
        root.put("payloadIncluded", true);
        root.put("runtimeLoadable", true);
        root.put("runtimeLoadMode", RUNTIME_LOAD_MODE);
        root.put("payloadAlignment", WdmlPackWriter.PAYLOAD_ALIGNMENT);
        root.put("payloadBytes", payloadBytes);

        Map<String, Object> encoderPayload = new LinkedHashMap<>();
        encoderPayload.put("format", "safetensors");
        encoderPayload.put("entryName", SAFETENSORS_SOURCE);
        encoderPayload.put("bytes", payloadBytes);
        root.put("encoderPayload", encoderPayload);

        SourceFingerprint fingerprint = SourceFingerprint.read(source);
        Map<String, Object> sourceInfo = new LinkedHashMap<>();
        sourceInfo.put("format", "safetensors");
        sourceInfo.put("fileName", source.getFileName().toString());
        sourceInfo.put("relativePath", source.getFileName().toString());
        sourceInfo.put("sizeBytes", fingerprint.sizeBytes());
        sourceInfo.put("lastModifiedMillis", fingerprint.lastModifiedMillis());
        sourceInfo.put("fileKey", fingerprint.fileKey());
        sourceInfo.put("fingerprint", fingerprint.value());
        root.put("source", sourceInfo);
        return root;
    }
}
