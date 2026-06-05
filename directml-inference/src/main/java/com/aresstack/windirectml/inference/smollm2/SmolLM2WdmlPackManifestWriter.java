package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.model.SourceTensor;
import com.aresstack.windirectml.inference.model.SourceTensorCatalog;
import com.aresstack.windirectml.inference.model.WdmlPackWriter;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Writes SmolLM2 wdmlpack packages.
 */
public final class SmolLM2WdmlPackManifestWriter {

    public Path writeManifestOnly(Path output,
                                  SmolLM2Config config,
                                  SmolLM2LayoutReport layoutReport) throws IOException {
        Objects.requireNonNull(output, "output");
        Map<String, Object> manifest = SmolLM2WdmlPackManifest.build(
                config,
                layoutReport,
                SmolLM2PayloadPolicy.MANIFEST_ONLY);
        return WdmlPackWriter.writeManifestOnly(output, manifest);
    }

    public Path writeWithDensePayload(Path output,
                                      SmolLM2Config config,
                                      SmolLM2LayoutReport layoutReport,
                                      SourceTensorCatalog catalog) throws IOException {
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(catalog, "catalog");
        PayloadPlan payloadPlan = planPayload(catalog);
        Map<String, Object> manifest = SmolLM2WdmlPackManifest.build(
                config,
                layoutReport,
                SmolLM2PayloadPolicy.DENSE_PAYLOAD,
                catalog,
                payloadPlan.tensorPlans());
        return WdmlPackWriter.writeWithPayload(output, manifest, payloadPlan.payloadEntries(), payloadPlan.payloadLength());
    }

    private static PayloadPlan planPayload(SourceTensorCatalog catalog) throws IOException {
        List<SmolLM2WdmlPackManifest.TensorPayloadPlan> tensorPlans = new ArrayList<>();
        List<WdmlPackWriter.PayloadEntry> payloadEntries = new ArrayList<>();
        long offset = 0L;
        for (SourceTensor tensor : catalog.entries().values()) {
            if (!tensor.hasPayload()) {
                throw new IOException("SmolLM2 tensor has no inline payload: " + tensor.name());
            }
            long currentOffset = offset;
            long byteLength = tensor.byteLength();
            tensorPlans.add(new SmolLM2WdmlPackManifest.TensorPayloadPlan(tensor.name(), currentOffset, byteLength));
            payloadEntries.add(new WdmlPackWriter.PayloadEntry(
                    tensor.name(),
                    currentOffset,
                    byteLength,
                    channel -> writeTensorPayload(channel, tensor.payloadBuffer())));
            offset += byteLength;
        }
        return new PayloadPlan(tensorPlans, payloadEntries, offset);
    }

    private static void writeTensorPayload(FileChannel channel, ByteBuffer payload) throws IOException {
        ByteBuffer buffer = payload.asReadOnlyBuffer();
        while (buffer.hasRemaining()) {
            channel.write(buffer);
        }
    }

    private record PayloadPlan(List<SmolLM2WdmlPackManifest.TensorPayloadPlan> tensorPlans,
                               List<WdmlPackWriter.PayloadEntry> payloadEntries,
                               long payloadLength) {
    }
}
