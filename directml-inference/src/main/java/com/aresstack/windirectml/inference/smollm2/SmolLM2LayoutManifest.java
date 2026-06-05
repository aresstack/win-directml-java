package com.aresstack.windirectml.inference.smollm2;

import java.util.Map;

/**
 * Manifest projection of a SmolLM2 layout report.
 */
public record SmolLM2LayoutManifest(SmolLM2LayoutReport report) {
    public Map<String, Object> toManifest() {
        return report.toManifest();
    }
}
