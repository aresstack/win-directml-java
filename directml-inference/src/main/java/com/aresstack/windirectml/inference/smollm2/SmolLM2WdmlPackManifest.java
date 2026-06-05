package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.model.WdmlPackWriter;

import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Builds the SmolLM2 wdmlpack manifest contract.
 */
public final class SmolLM2WdmlPackManifest {

    public static final int SCHEMA_VERSION = WdmlPackWriter.VERSION;
    public static final int COMPILER_VERSION = 35;

    private SmolLM2WdmlPackManifest() {
    }

    public static Map<String, Object> build(SmolLM2Config config,
                                            SmolLM2LayoutReport layoutReport,
                                            SmolLM2PayloadPolicy payloadPolicy) {
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("format", "wdmlpack");
        root.put("version", SCHEMA_VERSION);
        root.put("schemaVersion", SCHEMA_VERSION);
        root.put("compilerVersion", COMPILER_VERSION);
        root.put("createdAt", Instant.now().toString());
        root.put("modelFamily", "smollm2");
        root.put("architecture", "llama-causal-decoder");
        root.put("sourceFormat", "safetensors");
        root.put("payloadIncluded", payloadPolicy.payloadIncluded());
        root.put("runtimeLoadable", false);
        root.put("runtimeLoadMode", "unsupported");
        root.put("runtimeLoadableReason", SmolLM2LayoutReport.RUNTIME_NOT_IMPLEMENTED);
        root.put("layoutComplete", layoutReport.layoutComplete());
        root.put("model", SmolLM2PackageMetadata.from(config).toManifest());
        root.put("smollm2Layout", layoutReport.toManifest());
        root.put("tensorCatalog", tensorCatalog(layoutReport));
        return root;
    }

    private static Map<String, Object> tensorCatalog(SmolLM2LayoutReport layoutReport) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("count", layoutReport.foundTensorCount());
        out.put("knownTensorCount", layoutReport.knownTensorCount());
        out.put("unknownTensorCount", layoutReport.unknownTensorCount());
        return out;
    }
}
