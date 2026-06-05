package com.aresstack.windirectml.inference.model;

import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Format-neutral tensor catalog for model importers.
 *
 * <p>This is the import-side counterpart of the runtime tensor catalog. Model
 * families can analyze names, shapes, and payload availability without depending
 * on ONNX tensor classes.</p>
 */
public final class SourceTensorCatalog {

    private final Map<String, SourceTensor> entries;

    public SourceTensorCatalog(Collection<SourceTensor> entries) {
        Objects.requireNonNull(entries, "entries");
        LinkedHashMap<String, SourceTensor> byName = new LinkedHashMap<>();
        for (SourceTensor entry : entries) {
            if (entry != null && entry.name() != null && !entry.name().isBlank()) {
                byName.put(entry.name(), entry);
            }
        }
        this.entries = Collections.unmodifiableMap(byName);
    }

    public SourceTensor get(String name) {
        return entries.get(name);
    }

    public boolean contains(String name) {
        return entries.containsKey(name);
    }

    public Map<String, SourceTensor> entries() {
        return entries;
    }

    public int size() {
        return entries.size();
    }

    public long inlineBytes() {
        return sum(TensorStorageKind.INLINE);
    }

    public long externalBytes() {
        return sum(TensorStorageKind.EXTERNAL);
    }

    public long metadataOnlyCount() {
        return entries.values().stream()
                .filter(e -> e.storageKind() == TensorStorageKind.METADATA_ONLY)
                .count();
    }

    public TensorCatalog toTensorCatalog() {
        return new TensorCatalog(entries.values().stream()
                .map(SourceTensor::toTensorEntry)
                .toList());
    }

    public String summary() {
        return toTensorCatalog().summary();
    }

    private long sum(TensorStorageKind kind) {
        return entries.values().stream()
                .filter(e -> e.storageKind() == kind)
                .mapToLong(SourceTensor::byteLength)
                .sum();
    }
}
