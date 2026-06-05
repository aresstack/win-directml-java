package com.aresstack.windirectml.inference.model;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * Import source for restricted PyTorch state-dict checkpoints.
 *
 * <p>This class never executes Python pickle payloads. It uses
 * {@link TorchCheckpointInspector} to validate and interpret the checkpoint
 * metadata, then exposes tensor payloads as format-neutral {@link SourceTensor}
 * instances for model-family compilers. The runtime must still consume only
 * compiled {@code .wdmlpack} packages.</p>
 */
public final class TorchCheckpointModelSource implements ModelSource<SourceTensorCatalog> {

    public static final String FORMAT = "torch-state-dict";

    private final Path checkpoint;

    public TorchCheckpointModelSource(Path checkpoint) {
        this.checkpoint = Objects.requireNonNull(checkpoint, "checkpoint").toAbsolutePath().normalize();
    }

    public static TorchCheckpointModelSource of(Path checkpoint) {
        return new TorchCheckpointModelSource(checkpoint);
    }

    @Override
    public String format() {
        return FORMAT;
    }

    @Override
    public Path location() {
        return checkpoint;
    }

    @Override
    public SourceTensorCatalog load() throws IOException {
        if (!Files.isRegularFile(checkpoint)) {
            throw new IOException("Torch checkpoint does not exist: " + checkpoint);
        }
        TorchCheckpointInspection inspection = TorchCheckpointInspector.inspect(checkpoint);
        if (inspection.hasMissingStorageEntries()) {
            throw new IOException("Torch checkpoint has missing tensor storage entries: "
                    + inspection.missingStorageEntries().stream()
                    .map(TorchCheckpointTensor::storageEntryName)
                    .toList());
        }

        List<SourceTensor> sourceTensors = new ArrayList<>(inspection.tensorCount());
        try (ZipFile zip = new ZipFile(checkpoint.toFile())) {
            for (TorchCheckpointTensor tensor : inspection.tensors()) {
                sourceTensors.add(readTensor(zip, tensor));
            }
        }
        return new SourceTensorCatalog(sourceTensors);
    }

    private SourceTensor readTensor(ZipFile zip, TorchCheckpointTensor tensor) throws IOException {
        validateCompactTensor(tensor);
        if (tensor.tensorByteLength() > Integer.MAX_VALUE) {
            throw new IOException("Torch tensor is too large for current ByteBuffer-backed importer: "
                    + tensor.name() + " (" + tensor.tensorByteLength() + " bytes)");
        }

        ZipEntry storageEntry = zip.getEntry(tensor.storageEntryName());
        if (storageEntry == null) {
            throw new IOException("Torch tensor storage entry not found for " + tensor.name()
                    + ": " + tensor.storageEntryName());
        }
        byte[] storage = readAll(zip, storageEntry);
        long byteOffset = Math.multiplyExact(tensor.storageOffset(), tensor.dataType().bytesPerElement());
        long byteEnd = Math.addExact(byteOffset, tensor.tensorByteLength());
        if (byteOffset < 0 || byteEnd > storage.length) {
            throw new IOException("Torch tensor " + tensor.name() + " escapes storage entry "
                    + tensor.storageEntryName() + ": tensorRange=[" + byteOffset + ", " + byteEnd
                    + "] storageBytes=" + storage.length);
        }

        ByteBuffer payload = ByteBuffer.wrap(storage, Math.toIntExact(byteOffset), Math.toIntExact(tensor.tensorByteLength()))
                .slice()
                .order(ByteOrder.LITTLE_ENDIAN)
                .asReadOnlyBuffer();
        return SourceTensor.inline(tensor.name(), tensor.dataType(), tensor.shape(), tensor.tensorByteLength(), payload);
    }

    private static byte[] readAll(ZipFile zip, ZipEntry entry) throws IOException {
        try (InputStream in = zip.getInputStream(entry); ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            in.transferTo(out);
            return out.toByteArray();
        }
    }

    private static void validateCompactTensor(TorchCheckpointTensor tensor) throws IOException {
        long[] shape = tensor.shape();
        long[] stride = tensor.stride();
        if (stride.length != shape.length) {
            throw new IOException("Torch tensor " + tensor.name() + " has non-rank-matching stride: "
                    + tensor.strideText() + " for shape " + tensor.shapeText());
        }
        long expected = 1L;
        for (int index = shape.length - 1; index >= 0; index--) {
            long dim = shape[index];
            if (dim < 0) {
                throw new IOException("Torch tensor " + tensor.name() + " has negative dimension: " + dim);
            }
            long actualStride = stride[index];
            long expectedStride = dim == 0 ? 1L : expected;
            if (actualStride != expectedStride) {
                throw new IOException("Torch tensor " + tensor.name() + " is not compact row-major: shape="
                        + tensor.shapeText() + ", stride=" + tensor.strideText());
            }
            if (dim > 0) {
                expected = Math.multiplyExact(expected, dim);
            }
        }
    }
}
