package com.aresstack.windirectml.inference.model;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Objects;

/**
 * Stable source-file identity used by model package caches.
 *
 * <p>The fingerprint is intentionally based on cheap file-system metadata only.
 * It is not a cryptographic checksum and should only be used to reject stale
 * local cache packages before the runtime attempts to map payloads.</p>
 */
public final class SourceFingerprint {

    private final String fileName;
    private final long sizeBytes;
    private final long lastModifiedMillis;
    private final String fileKey;

    private SourceFingerprint(String fileName, long sizeBytes, long lastModifiedMillis, String fileKey) {
        this.fileName = fileName == null ? "" : fileName;
        this.sizeBytes = sizeBytes;
        this.lastModifiedMillis = lastModifiedMillis;
        this.fileKey = fileKey == null ? "" : fileKey;
    }

    public static SourceFingerprint read(Path source) {
        Path safeSource = source == null ? null : source.toAbsolutePath().normalize();
        String name = safeSource == null || safeSource.getFileName() == null
                ? ""
                : safeSource.getFileName().toString();
        return new SourceFingerprint(name, safeSize(safeSource), safeLastModifiedMillis(safeSource), readFileKey(safeSource));
    }

    public static String value(Path source, long sizeBytes, long lastModifiedMillis, String fileKey) {
        String name = source == null || source.getFileName() == null ? "" : source.getFileName().toString();
        return new SourceFingerprint(name, sizeBytes, lastModifiedMillis, fileKey).value();
    }

    public String fileName() {
        return fileName;
    }

    public long sizeBytes() {
        return sizeBytes;
    }

    public long lastModifiedMillis() {
        return lastModifiedMillis;
    }

    public String fileKey() {
        return fileKey;
    }

    public String value() {
        return fileName + "|" + sizeBytes + "|" + lastModifiedMillis + "|" + fileKey;
    }

    public boolean matches(String expected) {
        return Objects.equals(value(), expected);
    }

    public static String readFileKey(Path file) {
        if (file == null || !Files.exists(file)) {
            return "";
        }
        try {
            Object key = Files.readAttributes(file, BasicFileAttributes.class).fileKey();
            return key == null ? "" : key.toString();
        } catch (IOException | RuntimeException ignored) {
            return "";
        }
    }

    private static long safeSize(Path source) {
        try {
            return source != null && Files.isRegularFile(source) ? Files.size(source) : -1L;
        } catch (IOException | RuntimeException e) {
            return -1L;
        }
    }

    private static long safeLastModifiedMillis(Path source) {
        try {
            return source != null && Files.exists(source) ? Files.getLastModifiedTime(source).toMillis() : -1L;
        } catch (IOException | RuntimeException e) {
            return -1L;
        }
    }
}
