package com.aresstack.windirectml.inference.smollm2;

import java.nio.file.Path;

/**
 * Result of SmolLM2 model-directory analysis or manifest writing.
 */
public record SmolLM2CompileReport(
        Path output,
        boolean dryRun,
        boolean payloadIncluded,
        boolean runtimeLoadable,
        String runtimeLoadableReason,
        SmolLM2Config config,
        SmolLM2LayoutReport layoutReport
) {
}
