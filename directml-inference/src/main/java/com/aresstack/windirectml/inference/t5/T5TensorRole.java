package com.aresstack.windirectml.inference.t5;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Describes one validated T5 tensor role in the wdmlpack layout manifest.
 */
record T5TensorRole(String role,
                    String sourceName,
                    String runtimeName,
                    int dataType,
                    String dataTypeName,
                    long[] dims,
                    boolean required,
                    boolean tied) {
    T5TensorRole {
        dims = dims == null ? new long[0] : dims.clone();
    }

    static T5TensorRole tied(String role, String sourceName, String runtimeName) {
        return new T5TensorRole(role, sourceName, runtimeName, 0, "TIED", new long[0], true, true);
    }

    @Override
    public long[] dims() {
        return dims.clone();
    }

    Map<String, Object> toManifest() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("role", role);
        out.put("sourceName", sourceName);
        out.put("runtimeName", runtimeName);
        out.put("dataType", dataType);
        out.put("dataTypeName", dataTypeName);
        List<Long> shape = new ArrayList<>(dims.length);
        for (long dim : dims) {
            shape.add(dim);
        }
        out.put("dims", shape);
        out.put("required", required);
        out.put("tied", tied);
        return out;
    }
}
