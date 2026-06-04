package com.aresstack.windirectml.inference.t5;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Summarizes whether a SafeTensors source satisfies the curated T5 wdmlpack layout.
 */
record T5LayoutManifest(
        String schema,
        int compilerVersion,
        boolean safeTensorsSource,
        boolean complete,
        boolean runtimeLoadable,
        String runtimeLoadMode,
        int roleCount,
        int tensorCount,
        long payloadBytes,
        List<T5TensorRole> roles,
        List<String> missingRequired,
        List<String> shapeErrors,
        List<String> unsupportedRuntimeDtypes
) {
    static T5LayoutManifest notSafeTensors(String sourceFormat) {
        return new T5LayoutManifest(T5SafeTensorsLayoutCompiler.LAYOUT_SCHEMA,
                T5SafeTensorsLayoutCompiler.COMPILER_VERSION,
                false, false, false, "not-safetensors", 0, 0, 0L,
                List.of(), List.of("source is not SafeTensors: " + sourceFormat), List.of(), List.of());
    }

    Map<String, Object> toManifest() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("schema", schema);
        out.put("compilerVersion", compilerVersion);
        out.put("sourceLayout", "huggingface-t5-dense");
        out.put("safeTensorsSource", safeTensorsSource);
        out.put("complete", complete);
        out.put("runtimeLoadable", runtimeLoadable);
        out.put("runtimeLoadMode", runtimeLoadMode);
        out.put("roleCount", roleCount);
        out.put("tensorCount", tensorCount);
        out.put("payloadBytes", payloadBytes);
        out.put("missingRequired", missingRequired);
        out.put("shapeErrors", shapeErrors);
        out.put("unsupportedRuntimeDtypes", unsupportedRuntimeDtypes);
        out.put("roles", roles.stream().map(T5TensorRole::toManifest).toList());
        return out;
    }
}
