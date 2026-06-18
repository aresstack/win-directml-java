package com.aresstack.windirectml.inference.model;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Reads internal {@code .wdmlpack} containers.
 *
 * <p>This is the only class that maps the package file and interprets payload
 * offsets. {@link RuntimeModelPackage} remains a small descriptor while model
 * families consume {@link RuntimeTensorCatalog} instead of reading manifests
 * directly.</p>
 */
public final class WdmlPackReader {

    private WdmlPackReader() {
    }

    public static RuntimeModelPackage open(Path packagePath) throws IOException {
        Path normalized = Objects.requireNonNull(packagePath, "packagePath").toAbsolutePath().normalize();
        WdmlPackWriter.Header header = WdmlPackWriter.readHeader(normalized);
        WdmlPackManifest manifest = new WdmlPackManifest(WdmlPackWriter.readManifest(normalized));
        RuntimeModelPackage modelPackage = new RuntimeModelPackage(normalized, header, manifest, Files.size(normalized));
        manifest.validateRoot(header);
        return modelPackage;
    }

    public static RuntimeTensorCatalog mapPayloadTensors(RuntimeModelPackage modelPackage) throws IOException {
        Objects.requireNonNull(modelPackage, "modelPackage");
        if (!modelPackage.payloadIncluded()) {
            return RuntimeTensorCatalog.empty();
        }
        // Map each tensor's payload region individually with a long file offset, rather than mmap'ing the whole
        // container. This supports packages > 2 GB (e.g. the real Phi-3-mini ~2.39 GB package); each tensor region is
        // still < Integer.MAX_VALUE. The mapped regions remain valid after the channel is closed (the JVM keeps the
        // mapping alive until GC), matching how callers consume RuntimeTensor buffers after this method returns.
        try (FileChannel channel = FileChannel.open(modelPackage.packagePath(), StandardOpenOption.READ)) {
            List<RuntimeTensor> tensors = new ArrayList<>();
            for (java.util.Map<String, Object> tensor : modelPackage.manifestMetadata().requireListOfMaps("tensors")) {
                String name = WdmlPackManifest.stringValue(tensor.get("name"));
                int dataType = WdmlPackManifest.intValue(tensor.get("dataType"), 0);
                long[] dims = WdmlPackManifest.dimsValue(tensor.get("dims"));
                long byteLength = WdmlPackManifest.longValue(tensor.get("byteLength"), 0L);
                long payloadOffset = WdmlPackManifest.longValue(tensor.get("payloadOffset"), -1L);
                long payloadLength = WdmlPackManifest.longValue(tensor.get("payloadLength"), byteLength);
                if (name.isBlank()) {
                    continue;
                }
                if (payloadOffset >= 0 && payloadLength > 0) {
                    tensors.add(mapTensorPayload(channel, modelPackage, name, dims, dataType, payloadOffset, payloadLength));
                } else {
                    tensors.add(new RuntimeTensor(name, dims, dataType, ByteBuffer.allocate(0), 0));
                }
            }
            return new RuntimeTensorCatalog(tensors);
        }
    }

    private static RuntimeTensor mapTensorPayload(FileChannel channel,
                                                  RuntimeModelPackage modelPackage,
                                                  String name,
                                                  long[] dims,
                                                  int dataType,
                                                  long payloadOffset,
                                                  long payloadLength) throws IOException {
        WdmlPackWriter.Header header = modelPackage.header();
        long absoluteStart = header.payloadOffset() + payloadOffset;
        long absoluteEnd = absoluteStart + payloadLength;
        if (absoluteStart < header.payloadOffset() || absoluteEnd < absoluteStart
                || absoluteEnd > header.payloadOffset() + header.payloadLength()
                || absoluteEnd > modelPackage.fileSize() || payloadLength > Integer.MAX_VALUE) {
            throw new IOException("Invalid wdmlpack tensor payload range for " + name
                    + ": offset=" + payloadOffset + ", length=" + payloadLength);
        }
        // Long offset into the file; per-tensor length fits an int (validated above).
        MappedByteBuffer region = channel.map(FileChannel.MapMode.READ_ONLY, absoluteStart, payloadLength);
        region.order(ByteOrder.LITTLE_ENDIAN);
        return new RuntimeTensor(name, dims, dataType, region, (int) payloadLength);
    }
}
