package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.model.SourceFingerprint;
import com.aresstack.windirectml.inference.model.WdmlPackWriter;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Writes T5 wdmlpack packages used by the T5 runtime strand.
 */
final class T5WdmlPackManifestWriter {

    Path writeWithPayload(Path output,
                          Path modelDir,
                          T5ModelImport imported,
                          T5Config config,
                          T5LayoutManifest layout) throws IOException {
        PayloadPlan payloadPlan = planPayload(imported, layout);
        Map<String, Object> manifest = buildManifest(modelDir, imported, config, layout, payloadPlan);
        return WdmlPackWriter.writeWithPayload(output, manifest, payloadPlan.entries(), payloadPlan.payloadLength());
    }

    Path writeManifestOnly(Path output,
                           Path modelDir,
                           T5ModelImport imported,
                           T5Config config,
                           T5LayoutManifest layout) throws IOException {
        return WdmlPackWriter.writeManifestOnly(output, buildManifest(modelDir, imported, config, layout, null));
    }

    Map<String, Object> buildManifest(Path modelDir,
                                      T5ModelImport imported,
                                      T5Config config,
                                      T5LayoutManifest layout) throws IOException {
        return buildManifest(modelDir, imported, config, layout, null);
    }

    Map<String, Object> buildManifest(Path modelDir,
                                      T5ModelImport imported,
                                      T5Config config,
                                      T5LayoutManifest layout,
                                      PayloadPlan payloadPlan) throws IOException {
        Objects.requireNonNull(modelDir, "modelDir");
        Objects.requireNonNull(imported, "imported");
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(layout, "layout");

        boolean payloadIncluded = payloadPlan != null && payloadPlan.payloadLength() > 0;
        boolean weightsLoadable = payloadIncluded && layout.complete() && layout.unsupportedRuntimeDtypes().isEmpty();
        // T5RuntimePackage.open() builds T5Weights + the runtime structures whenever the weights load, so a
        // weights-loadable package IS runtime-loadable. Generation is not yet certified, so executable stays false
        // (T5-1: don't hide "not production-ready" behind runtimeLoadable=false).
        boolean runtimeLoadable = weightsLoadable;
        boolean executable = false;
        String runtimeLoadMode;
        String reason;
        if (!payloadIncluded) {
            runtimeLoadMode = T5ManifestPayloadPolicy.MODE_MANIFEST_ONLY;
            reason = T5ManifestPayloadPolicy.REASON_MANIFEST_ONLY;
        } else if (!weightsLoadable) {
            runtimeLoadMode = T5ManifestPayloadPolicy.MODE_WEIGHTS_NOT_LOADABLE;
            reason = T5ManifestPayloadPolicy.REASON_WEIGHTS_NOT_LOADABLE;
        } else {
            runtimeLoadMode = T5ManifestPayloadPolicy.MODE_RUNTIME_LOADABLE_NOT_EXECUTABLE;
            reason = T5ManifestPayloadPolicy.REASON_RUNTIME_LOADABLE_NOT_EXECUTABLE;
        }

        Map<String, Object> root = new LinkedHashMap<>();
        root.put("format", "wdmlpack");
        root.put("version", WdmlPackWriter.VERSION);
        root.put("createdAt", Instant.now().toString());
        root.put("modelFamily", T5PackageMetadata.MODEL_FAMILY);
        root.put("architecture", T5PackageMetadata.ARCHITECTURE);
        root.put("sourceFormat", imported.sourceFormat());
        root.put("sourceLayout", layout.sourceLayout());
        root.put("mode", payloadIncluded ? "payload" : "manifest-only");
        root.put("payloadIncluded", payloadIncluded);
        root.put("weightsLoadable", weightsLoadable);
        root.put("runtimeLoadable", runtimeLoadable);
        root.put("executable", executable);
        root.put("runtimeLoadMode", runtimeLoadMode);
        root.put("reason", reason);
        if (payloadIncluded) {
            root.put("payloadAlignment", WdmlPackWriter.PAYLOAD_ALIGNMENT);
            root.put("payloadBytes", payloadPlan.payloadLength());
        }
        root.put("source", buildSource(modelDir, imported));
        root.put("model", buildModelSection(config));
        root.put("t5", T5PackageMetadata.from(config).toManifest());
        root.put("layout", layout.toManifest());
        root.put("tensors", buildTensorDirectory(imported, layout, payloadPlan));
        return root;
    }

    private PayloadPlan planPayload(T5ModelImport imported, T5LayoutManifest layout) throws IOException {
        List<T5TensorRole> roles = new ArrayList<>(layout.roles());
        roles.sort(Comparator.comparing(T5TensorRole::runtimeName));
        Map<String, Long> offsets = new LinkedHashMap<>();
        Map<String, Long> lengths = new LinkedHashMap<>();
        List<WdmlPackWriter.PayloadEntry> entries = new ArrayList<>();
        long cursor = 0;
        for (T5TensorRole role : roles) {
            if (role.tied()) {
                offsets.put(role.runtimeName(), -1L);
                lengths.put(role.runtimeName(), 0L);
                continue;
            }
            OnnxTensor tensor = imported.inlineTensors().get(role.sourceName());
            if (tensor == null || tensor.rawByteLength() <= 0) {
                offsets.put(role.runtimeName(), -1L);
                lengths.put(role.runtimeName(), 0L);
                continue;
            }
            cursor = WdmlPackWriter.align(cursor, 64);
            long offset = cursor;
            long length = tensor.rawByteLength();
            offsets.put(role.runtimeName(), offset);
            lengths.put(role.runtimeName(), length);
            entries.add(new WdmlPackWriter.PayloadEntry(role.runtimeName(), offset, length,
                    channel -> writeTensor(channel, tensor)));
            cursor += length;
        }
        return new PayloadPlan(offsets, lengths, entries, cursor);
    }

    private static void writeTensor(FileChannel channel, OnnxTensor tensor) throws IOException {
        ByteBuffer data = tensor.rawDataBuffer().order(ByteOrder.LITTLE_ENDIAN);
        data.position(0);
        while (data.hasRemaining()) {
            channel.write(data);
        }
    }

    private List<Map<String, Object>> buildTensorDirectory(T5ModelImport imported,
                                                           T5LayoutManifest layout,
                                                           PayloadPlan payloadPlan) {
        List<T5TensorRole> roles = new ArrayList<>(layout.roles());
        roles.sort(Comparator.comparing(T5TensorRole::runtimeName));
        List<Map<String, Object>> tensors = new ArrayList<>(roles.size());
        for (T5TensorRole role : roles) {
            Map<String, Object> tensor = role.toManifest();
            tensor.put("name", role.runtimeName());
            long byteLength = payloadPlan == null ? sourceLength(imported, role) : payloadPlan.lengths().getOrDefault(role.runtimeName(), 0L);
            long payloadOffset = payloadPlan == null ? -1L : payloadPlan.offsets().getOrDefault(role.runtimeName(), -1L);
            tensor.put("byteLength", byteLength);
            tensor.put("payloadOffset", payloadOffset);
            tensor.put("payloadLength", payloadOffset >= 0 ? byteLength : 0L);
            tensors.add(tensor);
        }
        return tensors;
    }

    private static long sourceLength(T5ModelImport imported, T5TensorRole role) {
        OnnxTensor tensor = imported.inlineTensors().get(role.sourceName());
        return tensor == null ? 0L : tensor.rawByteLength();
    }

    private Map<String, Object> buildSource(Path modelDir, T5ModelImport imported) throws IOException {
        Map<String, Object> source = new LinkedHashMap<>();
        Path root = modelDir.toAbsolutePath().normalize();
        Path sourcePath = imported.modelPath().toAbsolutePath().normalize();
        source.put("format", imported.sourceFormat());
        source.put("fileName", sourcePath.getFileName().toString());
        source.put("relativePath", sourcePath.startsWith(root) ? root.relativize(sourcePath).toString().replace('\\', '/') : sourcePath.getFileName().toString());
        if (java.nio.file.Files.isRegularFile(sourcePath)) {
            SourceFingerprint fingerprint = SourceFingerprint.read(sourcePath);
            source.put("sizeBytes", fingerprint.sizeBytes());
            source.put("lastModifiedMillis", fingerprint.lastModifiedMillis());
            source.put("fileKey", fingerprint.fileKey());
            source.put("fingerprint", fingerprint.value());
        }
        return source;
    }

    private Map<String, Object> buildModelSection(T5Config config) {
        Map<String, Object> model = new LinkedHashMap<>();
        model.put("architecture", T5PackageMetadata.ARCHITECTURE);
        model.put("modelFamily", T5PackageMetadata.MODEL_FAMILY);
        model.put("dModel", config.modelSize());
        model.put("dKv", config.keyValueSize());
        model.put("dFf", config.feedForwardSize());
        model.put("numHeads", config.attentionHeads());
        model.put("encoderLayers", config.encoderLayers());
        model.put("decoderLayers", config.effectiveDecoderLayers());
        model.put("vocabSize", config.vocabSize());
        model.put("feedForwardProjection", config.effectiveFeedForwardProjection());
        model.put("tieWordEmbeddings", config.usesTiedWordEmbeddings());
        return model;
    }

    record PayloadPlan(Map<String, Long> offsets,
                       Map<String, Long> lengths,
                       List<WdmlPackWriter.PayloadEntry> entries,
                       long payloadLength) {
    }
}
