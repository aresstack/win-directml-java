package com.aresstack.windirectml.inference.artifact;

import java.nio.file.Path;
import java.util.Objects;

/**
 * The conversion the lifecycle would perform for a model in its current state. Produced by
 * {@code planConversion()}; computing a plan never writes anything.
 *
 * @param action            the state-dependent action (drives the button label)
 * @param targetPackagePath where {@code convert()} would write the package, or {@code null}
 *                          when the action does not write a package
 * @param sourceDescription a short human description of the raw source that would be compiled
 * @param reason            why this action was chosen
 */
public record ModelConversionPlan(ModelConversionAction action,
                                  Path targetPackagePath,
                                  String sourceDescription,
                                  String reason) {

    public ModelConversionPlan {
        action = Objects.requireNonNull(action, "action");
        sourceDescription = sourceDescription == null ? "" : sourceDescription;
        reason = reason == null ? "" : reason;
    }
}
