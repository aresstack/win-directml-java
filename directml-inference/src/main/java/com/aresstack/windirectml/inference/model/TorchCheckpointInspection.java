package com.aresstack.windirectml.inference.model;

import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Inspection result for a restricted PyTorch state-dict checkpoint scan.
 */
public record TorchCheckpointInspection(
        Path checkpoint,
        String archivePrefix,
        int tensorCount,
        long declaredTensorBytes,
        long storageBytes,
        List<TorchCheckpointTensor> tensors
) {
    public TorchCheckpointInspection {
        checkpoint = checkpoint == null ? null : checkpoint.toAbsolutePath().normalize();
        archivePrefix = archivePrefix == null ? "" : archivePrefix;
        tensors = Collections.unmodifiableList(List.copyOf(Objects.requireNonNull(tensors, "tensors")));
    }

    public boolean hasMissingStorageEntries() {
        return tensors.stream().anyMatch(tensor -> !tensor.storageEntryPresent());
    }

    public List<TorchCheckpointTensor> missingStorageEntries() {
        return tensors.stream()
                .filter(tensor -> !tensor.storageEntryPresent())
                .toList();
    }
}
