package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.model.RuntimeModelPackage;
import com.aresstack.windirectml.inference.model.SafeTensorsReader;
import com.aresstack.windirectml.inference.model.WdmlPackWriter;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
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

    /**
     * Heap-light WARP weights for the product runtime (GEMMA-WARP-13a): mmaps the SafeTensors payload and
     * decodes each layer projection and the tied embedding/LM head into direct FP32 {@link ByteBuffer}s
     * (off-heap), so the JVM heap no longer carries the ~1 GB of weights that
     * {@link #loadReferenceWeights()} materialises as {@code float[]}. Norm vectors stay {@code float[]}
     * (small). No format change — same payload, different decode target.
     */
    public Gemma3WarpWeights loadWarpWeightsHeapLight() throws IOException {
        WdmlPackWriter.Header header = modelPackage.header();
        if (!header.payloadIncluded()) {
            throw new IOException("Gemma wdmlpack has no weight payload: " + packagePath);
        }
        try (FileChannel channel = FileChannel.open(packagePath, StandardOpenOption.READ)) {
            MappedByteBuffer payload = channel.map(
                    FileChannel.MapMode.READ_ONLY, header.payloadOffset(), header.payloadLength());
            SafeTensorsReader.SafeTensorsFile file = SafeTensorsReader.readFromBuffer(packagePath, payload);

            ByteBuffer embedding = Gemma3WeightBufferView.decodeFp32LittleEndian(
                    entry(file, Gemma3TensorNameMapper.EMBED_TOKENS));
            float[] finalNorm = Gemma3ReferenceWeights.decodeFloats(entry(file, Gemma3TensorNameMapper.FINAL_NORM));

            int n = config.numHiddenLayers();
            Gemma3WarpLayerWeights[] layers = new Gemma3WarpLayerWeights[n];
            for (int i = 0; i < n; i++) {
                layers[i] = Gemma3WarpLayerWeights.ofByteBufferProjections(
                        floats(file, Gemma3TensorNameMapper.inputLayerNorm(i)),
                        buffer(file, Gemma3TensorNameMapper.qProj(i)),
                        buffer(file, Gemma3TensorNameMapper.kProj(i)),
                        buffer(file, Gemma3TensorNameMapper.vProj(i)),
                        buffer(file, Gemma3TensorNameMapper.oProj(i)),
                        floats(file, Gemma3TensorNameMapper.qNorm(i)),
                        floats(file, Gemma3TensorNameMapper.kNorm(i)),
                        floats(file, Gemma3TensorNameMapper.postAttentionLayerNorm(i)),
                        floats(file, Gemma3TensorNameMapper.preFeedforwardLayerNorm(i)),
                        buffer(file, Gemma3TensorNameMapper.gateProj(i)),
                        buffer(file, Gemma3TensorNameMapper.upProj(i)),
                        buffer(file, Gemma3TensorNameMapper.downProj(i)),
                        floats(file, Gemma3TensorNameMapper.postFeedforwardLayerNorm(i)));
            }
            return Gemma3WarpWeights.ofByteBufferEmbedding(config, embedding, finalNorm, layers);
        }
    }

    private static SafeTensorsReader.SafeTensorEntry entry(
            SafeTensorsReader.SafeTensorsFile file, String name) throws IOException {
        SafeTensorsReader.SafeTensorEntry e = file.tensors().get(name);
        if (e == null) {
            throw new IOException("Gemma wdmlpack payload is missing tensor: " + name);
        }
        return e;
    }

    private static ByteBuffer buffer(SafeTensorsReader.SafeTensorsFile file, String name) throws IOException {
        return Gemma3WeightBufferView.decodeFp32LittleEndian(entry(file, name));
    }

    private static float[] floats(SafeTensorsReader.SafeTensorsFile file, String name) throws IOException {
        return Gemma3ReferenceWeights.decodeFloats(entry(file, name));
    }
}
