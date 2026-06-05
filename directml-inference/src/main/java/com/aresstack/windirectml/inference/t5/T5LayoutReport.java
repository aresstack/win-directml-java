package com.aresstack.windirectml.inference.t5;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Human-readable T5 layout report used by CLI output and manifest serialization.
 */
final class T5LayoutReport {
    private final T5LayoutManifest manifest;

    T5LayoutReport(T5LayoutManifest manifest) {
        this.manifest = Objects.requireNonNull(manifest, "manifest");
    }

    T5LayoutManifest manifest() {
        return manifest;
    }

    Map<String, Object> toManifest() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("schema", manifest.schema());
        out.put("compilerVersion", manifest.compilerVersion());
        out.put("sourceLayout", manifest.sourceLayout());
        out.put("safeTensorsSource", manifest.safeTensorsSource());
        out.put("complete", manifest.complete());
        out.put("runtimeLoadable", manifest.runtimeLoadable());
        out.put("runtimeLoadMode", manifest.runtimeLoadMode());
        out.put("reason", manifest.reason());
        out.put("roleCount", manifest.roleCount());
        out.put("tensorCount", manifest.tensorCount());
        out.put("payloadBytes", manifest.payloadBytes());
        out.put("missingRequired", manifest.missingRequired());
        out.put("shapeErrors", manifest.shapeErrors());
        out.put("unsupportedRuntimeDtypes", manifest.unsupportedRuntimeDtypes());
        out.put("roles", manifest.roles().stream().map(T5TensorRole::toManifest).toList());
        return out;
    }

    List<String> missingRequired() {
        return manifest.missingRequired();
    }
}
