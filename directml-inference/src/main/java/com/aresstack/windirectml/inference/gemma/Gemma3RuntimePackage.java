package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.model.RuntimeModelPackage;
import com.aresstack.windirectml.inference.model.SafeTensorsReader;
import com.aresstack.windirectml.inference.model.WdmlPackWriter;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.Objects;

/**
 * Gemma 3 view over a {@code .wdmlpack} produced by {@link Gemma3WdmlPackCompiler}. Validates the
 * family manifest, exposes the {@link Gemma3Config}, and loads the CPU reference weights from the
 * embedded SafeTensors payload (the native runtime loads only from this package, not from raw HF files).
 */
public final class Gemma3RuntimePackage {

    private final Path packagePath;
    private final RuntimeModelPackage modelPackage;
    private final Gemma3Config config;

    private Gemma3RuntimePackage(Path packagePath, RuntimeModelPackage modelPackage, Gemma3Config config) {
        this.packagePath = packagePath;
        this.modelPackage = modelPackage;
        this.config = config;
    }

    public static Gemma3RuntimePackage open(Path packagePath) throws IOException {
        RuntimeModelPackage pkg = RuntimeModelPackage.open(packagePath);
        Map<String, Object> manifest = pkg.manifest();
        if (!Gemma3WdmlPackCompiler.MODEL_FAMILY.equals(String.valueOf(manifest.get("modelFamily")))) {
            throw new IOException("Not a Gemma 3 wdmlpack: modelFamily=" + manifest.get("modelFamily"));
        }
        Object configJson = manifest.get("gemmaConfigJson");
        if (!(configJson instanceof String json) || json.isBlank()) {
            throw new IOException("Gemma wdmlpack manifest is missing gemmaConfigJson");
        }
        Gemma3Config config = new Gemma3ConfigReader().read(json);
        return new Gemma3RuntimePackage(packagePath, pkg, config);
    }

    public Gemma3Config config() {
        return config;
    }

    public boolean runtimeLoadable() {
        return modelPackage.runtimeLoadable();
    }

    public String runtimeLoadMode() {
        return modelPackage.runtimeLoadMode();
    }

    /** Load the CPU reference weights from the embedded SafeTensors payload (decoded F32/F16/BF16). */
    public Gemma3ReferenceWeights loadReferenceWeights() throws IOException {
        WdmlPackWriter.Header header = modelPackage.header();
        if (!header.payloadIncluded()) {
            throw new IOException("Gemma wdmlpack has no weight payload: " + packagePath);
        }
        byte[] image = new byte[Math.toIntExact(header.payloadLength())];
        try (FileChannel channel = FileChannel.open(packagePath, StandardOpenOption.READ)) {
            ByteBuffer buffer = ByteBuffer.wrap(image);
            while (buffer.hasRemaining()) {
                int read = channel.read(buffer, header.payloadOffset() + buffer.position());
                if (read < 0) {
                    throw new IOException("Truncated Gemma wdmlpack payload: " + packagePath);
                }
            }
        }
        SafeTensorsReader.SafeTensorsFile file = SafeTensorsReader.readFromBuffer(
                Objects.requireNonNull(packagePath), ByteBuffer.wrap(image));
        return Gemma3ReferenceWeights.load(file, config);
    }
}
