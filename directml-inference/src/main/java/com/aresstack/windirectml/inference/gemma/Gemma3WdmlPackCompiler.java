package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.model.SafeTensorsReader;
import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorEntry;
import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorsFile;
import com.aresstack.windirectml.inference.model.SourceFingerprint;
import com.aresstack.windirectml.inference.model.WdmlPackWriter;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Compiles a Hugging Face Gemma 3 (text) model directory ({@code config.json} + {@code model.safetensors})
 * into a {@code .wdmlpack}, so the runtime loads from a package like the other families instead of from
 * raw HF files. Validates that every required tensor is present with the expected shape and a supported
 * dtype (F32/F16/BF16) before writing.
 *
 * <p>The payload is the verbatim SafeTensors image (parity by construction with the proven reference);
 * a native per-tensor Gemma layout is a later optimization. {@code runtimeLoadMode} is
 * {@code gemma3-safetensors-payload}.</p>
 */
public final class Gemma3WdmlPackCompiler {

    public static final String DEFAULT_OUTPUT_NAME = "model_gemma3.wdmlpack";
    public static final String MODEL_FAMILY = "gemma3";
    public static final String ARCHITECTURE = "gemma3-causal-decoder";
    public static final String RUNTIME_LOAD_MODE = "gemma3-safetensors-payload";
    private static final String SAFETENSORS_SOURCE = "model.safetensors";

    private Gemma3WdmlPackCompiler() {
    }

    public record CompileResult(Path output, boolean written, boolean runtimeLoadable,
                                int tensorCount, long payloadBytes,
                                List<String> missing, List<String> shapeErrors, List<String> dtypeErrors) {
        public CompileResult {
            missing = List.copyOf(missing);
            shapeErrors = List.copyOf(shapeErrors);
            dtypeErrors = List.copyOf(dtypeErrors);
        }
    }

    public static CompileResult compile(Path modelDir, Path output, boolean force) throws IOException {
        Path dir = modelDir.toAbsolutePath().normalize();
        Path configPath = dir.resolve("config.json");
        Path safetensors = dir.resolve(SAFETENSORS_SOURCE);
        if (!Files.isRegularFile(configPath)) {
            throw new IOException("Missing Gemma config.json in " + dir);
        }
        if (!Files.isRegularFile(safetensors)) {
            throw new IOException("Missing " + SAFETENSORS_SOURCE + " in " + dir);
        }
        String configJson = Files.readString(configPath);
        Gemma3Config config = new Gemma3ConfigReader().read(configJson);
        SafeTensorsFile file = SafeTensorsReader.read(safetensors);

        Layout layout = validateLayout(config, file);
        boolean runtimeLoadable = layout.ok();

        Path out = output == null ? dir.resolve(DEFAULT_OUTPUT_NAME) : output.toAbsolutePath().normalize();
        if (Files.exists(out) && !force) {
            throw new IOException("Output already exists: " + out + " (use force to overwrite)");
        }
        byte[] image = Files.readAllBytes(safetensors);
        Map<String, Object> manifest = buildManifest(config, configJson, safetensors, image.length,
                runtimeLoadable, file.tensors().size());
        List<WdmlPackWriter.PayloadEntry> entries = List.of(new WdmlPackWriter.PayloadEntry(
                SAFETENSORS_SOURCE, 0L, image.length, channel -> {
            ByteBuffer b = ByteBuffer.wrap(image);
            while (b.hasRemaining()) {
                channel.write(b);
            }
        }));
        WdmlPackWriter.writeWithPayload(out, manifest, entries, image.length);
        return new CompileResult(out, true, runtimeLoadable, file.tensors().size(), image.length,
                layout.missing, layout.shapeErrors, layout.dtypeErrors);
    }

    private record Layout(List<String> missing, List<String> shapeErrors, List<String> dtypeErrors) {
        boolean ok() {
            return missing.isEmpty() && shapeErrors.isEmpty() && dtypeErrors.isEmpty();
        }
    }

    private static Layout validateLayout(Gemma3Config config, SafeTensorsFile file) {
        List<String> missing = new ArrayList<>();
        List<String> shapeErrors = new ArrayList<>();
        List<String> dtypeErrors = new ArrayList<>();
        Map<String, long[]> expected = Gemma3TensorNameMapper.expectedShapes(config);
        for (Map.Entry<String, long[]> e : expected.entrySet()) {
            SafeTensorEntry t = file.tensors().get(e.getKey());
            if (t == null) {
                missing.add(e.getKey());
                continue;
            }
            if (!Arrays.equals(t.shape(), e.getValue())) {
                shapeErrors.add(e.getKey() + " expected " + Arrays.toString(e.getValue())
                        + " got " + Arrays.toString(t.shape()));
            }
            String dt = t.dtype();
            if (!dt.equals("F32") && !dt.equals("F16") && !dt.equals("BF16")) {
                dtypeErrors.add(e.getKey() + " unsupported dtype " + dt);
            }
        }
        return new Layout(missing, shapeErrors, dtypeErrors);
    }

    private static Map<String, Object> buildManifest(Gemma3Config config, String configJson, Path source,
                                                     long payloadBytes, boolean runtimeLoadable, int tensorCount)
            throws IOException {
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("format", "wdmlpack");
        root.put("version", WdmlPackWriter.VERSION);
        root.put("createdAt", Instant.now().toString());
        root.put("modelFamily", MODEL_FAMILY);
        root.put("architecture", ARCHITECTURE);
        root.put("sourceFormat", "safetensors");
        root.put("payloadIncluded", true);
        root.put("runtimeLoadable", runtimeLoadable);
        root.put("runtimeLoadMode", RUNTIME_LOAD_MODE);
        root.put("payloadAlignment", WdmlPackWriter.PAYLOAD_ALIGNMENT);
        root.put("payloadBytes", payloadBytes);
        root.put("gemmaConfigJson", configJson);

        Map<String, Object> gemma = new LinkedHashMap<>();
        gemma.put("hiddenSize", config.hiddenSize());
        gemma.put("numHiddenLayers", config.numHiddenLayers());
        gemma.put("numAttentionHeads", config.numAttentionHeads());
        gemma.put("numKeyValueHeads", config.numKeyValueHeads());
        gemma.put("headDim", config.headDim());
        gemma.put("vocabSize", config.vocabSize());
        gemma.put("slidingWindow", config.slidingWindow());
        gemma.put("tensorCount", tensorCount);
        root.put("gemma", gemma);

        SourceFingerprint fp = SourceFingerprint.read(source);
        Map<String, Object> src = new LinkedHashMap<>();
        src.put("format", "safetensors");
        src.put("fileName", source.getFileName().toString());
        src.put("relativePath", source.getFileName().toString());
        src.put("sizeBytes", fp.sizeBytes());
        src.put("lastModifiedMillis", fp.lastModifiedMillis());
        src.put("fileKey", fp.fileKey());
        src.put("fingerprint", fp.value());
        root.put("source", src);
        return root;
    }
}
