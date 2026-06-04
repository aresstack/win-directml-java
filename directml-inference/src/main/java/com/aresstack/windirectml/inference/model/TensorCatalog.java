package com.aresstack.windirectml.inference.model;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * A compact, format-neutral tensor directory built by the import layer.
 *
 * <p>The runtime can use this as the stable seam for future formats and later
 * for a flat {@code .wdmlpack} cache. It intentionally stores only metadata,
 * not large Java heap payloads.</p>
 */
public final class TensorCatalog {

    private final Map<String, TensorEntry> entries;

    public TensorCatalog(Collection<TensorEntry> entries) {
        Objects.requireNonNull(entries, "entries");
        LinkedHashMap<String, TensorEntry> byName = new LinkedHashMap<>();
        for (TensorEntry entry : entries) {
            if (entry != null && entry.name() != null && !entry.name().isBlank()) {
                byName.put(entry.name(), entry);
            }
        }
        this.entries = Map.copyOf(byName);
    }

    public TensorEntry get(String name) {
        return entries.get(name);
    }

    public Map<String, TensorEntry> entries() {
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

    private long sum(TensorStorageKind kind) {
        return entries.values().stream()
                .filter(e -> e.storageKind() == kind)
                .mapToLong(TensorEntry::byteLength)
                .sum();
    }

    public String summary() {
        return "tensors=" + size()
                + ", inline=" + formatBytes(inlineBytes())
                + ", external=" + formatBytes(externalBytes())
                + ", metadataOnly=" + metadataOnlyCount();
    }

    private static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        double kb = bytes / 1024.0;
        if (kb < 1024) return String.format(java.util.Locale.ROOT, "%.1f KiB", kb);
        double mb = kb / 1024.0;
        if (mb < 1024) return String.format(java.util.Locale.ROOT, "%.1f MiB", mb);
        double gb = mb / 1024.0;
        return String.format(java.util.Locale.ROOT, "%.2f GiB", gb);
    }
}
