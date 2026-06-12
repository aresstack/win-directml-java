package com.aresstack.windirectml.inference.artifact;

import java.nio.file.Path;

/**
 * Central, device-free entry point for the unified model artifact lifecycle. This is the stable
 * surface the Workbench (and later the release library) uses to inspect, plan and convert model
 * packages.
 *
 * <p>Strict contract:</p>
 * <ul>
 *   <li>{@link #inspect} never compiles;</li>
 *   <li>{@link #validateOrThrowBeforeInference} never compiles;</li>
 *   <li>{@link #convert} is the only method that may write a {@code .wdmlpack}.</li>
 * </ul>
 */
public interface ModelArtifactService {

    /** Inspect raw + package state for a model. Never writes/compiles. */
    ModelArtifactStatus inspect(ModelFamily family, Path modelDir);

    /** Plan the state-dependent conversion action. Never writes/compiles. */
    ModelConversionPlan planConversion(ModelFamily family, Path modelDir);

    /** Compile the raw source into a runtime package. The only write path. */
    ModelConversionResult convert(ModelFamily family, Path modelDir, boolean force);

    /** Throw an actionable error if inference must not run for this model. Never compiles. */
    void validateOrThrowBeforeInference(ModelFamily family, Path modelDir);

    /** The lifecycle adapter for a family (for callers that need direct access). */
    ModelPackageLifecycle lifecycle(ModelFamily family);
}
