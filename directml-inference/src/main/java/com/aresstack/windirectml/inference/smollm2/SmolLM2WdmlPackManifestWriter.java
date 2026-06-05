package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.model.WdmlPackWriter;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;

/**
 * Writes manifest-only SmolLM2 wdmlpack packages.
 */
public final class SmolLM2WdmlPackManifestWriter {

    public Path writeManifestOnly(Path output,
                                  SmolLM2Config config,
                                  SmolLM2LayoutReport layoutReport) throws IOException {
        Objects.requireNonNull(output, "output");
        Map<String, Object> manifest = SmolLM2WdmlPackManifest.build(
                config,
                layoutReport,
                SmolLM2PayloadPolicy.MANIFEST_ONLY);
        return WdmlPackWriter.writeManifestOnly(output, manifest);
    }
}
