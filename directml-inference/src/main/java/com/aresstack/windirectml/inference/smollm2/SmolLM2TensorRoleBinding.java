package com.aresstack.windirectml.inference.smollm2;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Binds a source tensor name to a SmolLM2 role and optional layer index.
 */
public record SmolLM2TensorRoleBinding(SmolLM2TensorRole role, int layerIndex, String tensorName) {
    public static final int NO_LAYER = -1;

    public SmolLM2TensorRoleBinding {
        role = Objects.requireNonNull(role, "role");
        if (role.layerBound() && layerIndex < 0) {
            throw new IllegalArgumentException("layerIndex must be >= 0 for layer-bound role " + role);
        }
        if (!role.layerBound()) {
            layerIndex = NO_LAYER;
        }
        if (tensorName == null || tensorName.isBlank()) {
            throw new IllegalArgumentException("tensorName must not be blank");
        }
    }

    public String key() {
        return role.layerBound() ? role.name() + "#" + layerIndex : role.name();
    }

    public Map<String, Object> toManifest() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("role", role.name());
        if (role.layerBound()) {
            out.put("layerIndex", layerIndex);
        }
        out.put("tensorName", tensorName);
        return out;
    }
}
