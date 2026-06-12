package com.aresstack.windirectml.inference.artifact;

import java.nio.file.Path;

/**
 * Outcome of a {@code convert()} call - the single artifact operation allowed to write a package.
 *
 * @param ok      whether the resulting package is runtime-loadable/executable
 * @param output  the package file that was written, or {@code null} when nothing was written
 * @param message a human-readable summary
 */
public record ModelConversionResult(boolean ok, Path output, String message) {

    public ModelConversionResult {
        message = message == null ? "" : message;
    }

    public static ModelConversionResult failed(String message) {
        return new ModelConversionResult(false, null, message);
    }
}
