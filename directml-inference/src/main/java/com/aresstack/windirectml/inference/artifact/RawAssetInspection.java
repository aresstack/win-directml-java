package com.aresstack.windirectml.inference.artifact;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Device-free, filesystem-only inspection of a model's raw source files. Mirrors the
 * differentiated states of {@code ModelAssetValidation} (missing / incomplete / corrupt /
 * valid) but lives in {@code directml-inference} so the artifact lifecycle has no cross-module
 * dependency. Performs no runtime work and never writes.
 */
public final class RawAssetInspection {

    private RawAssetInspection() {
    }

    /** Result of a raw inspection: the state plus a human-readable detail for the status reason. */
    public record Result(RawAssetState state, String detail) {
    }

    /**
     * Inspect {@code modelDir} for every name in {@code requiredFiles}. A directory is
     * {@link RawAssetState#RAW_VALID} only when each required file exists as a non-empty regular
     * file. {@code anyOfGroups} expresses "at least one of these alternatives must be present and
     * non-empty" (e.g. {@code *.safetensors} OR {@code pytorch_model.bin}); pass an empty list when
     * not needed.
     */
    public static Result inspect(Path modelDir, List<String> requiredFiles, List<List<String>> anyOfGroups) {
        if (modelDir == null || !Files.isDirectory(modelDir)) {
            return new Result(RawAssetState.RAW_MISSING, "model directory not found: " + modelDir);
        }
        List<String> missing = new ArrayList<>();
        List<String> empty = new ArrayList<>();
        for (String file : requiredFiles) {
            Path candidate = modelDir.resolve(file);
            if (!Files.isRegularFile(candidate)) {
                missing.add(file);
            } else if (isZeroBytes(candidate)) {
                empty.add(file);
            }
        }
        for (List<String> group : anyOfGroups) {
            if (!anyPresentNonEmpty(modelDir, group)) {
                missing.add("one of " + group);
            }
        }
        if (!missing.isEmpty()) {
            return new Result(RawAssetState.RAW_INCOMPLETE,
                    "incomplete download: missing required file(s) " + missing);
        }
        if (!empty.isEmpty()) {
            return new Result(RawAssetState.RAW_CORRUPT,
                    "corrupt (zero-byte) source file(s) " + empty);
        }
        return new Result(RawAssetState.RAW_VALID, "all required source files present");
    }

    /** Whether the directory contains at least one non-empty regular file ending with {@code suffix}. */
    public static boolean hasFileWithSuffix(Path modelDir, String suffix) {
        if (modelDir == null || !Files.isDirectory(modelDir)) {
            return false;
        }
        try (var stream = Files.list(modelDir)) {
            return stream.filter(Files::isRegularFile)
                    .anyMatch(p -> p.getFileName().toString().endsWith(suffix) && !isZeroBytes(p));
        } catch (IOException e) {
            return false;
        }
    }

    private static boolean anyPresentNonEmpty(Path modelDir, List<String> names) {
        for (String name : names) {
            if (name.startsWith("*")) {
                if (hasFileWithSuffix(modelDir, name.substring(1))) {
                    return true;
                }
                continue;
            }
            Path candidate = modelDir.resolve(name);
            if (Files.isRegularFile(candidate) && !isZeroBytes(candidate)) {
                return true;
            }
        }
        return false;
    }

    private static boolean isZeroBytes(Path file) {
        try {
            return Files.size(file) == 0L;
        } catch (IOException e) {
            return false;
        }
    }
}
