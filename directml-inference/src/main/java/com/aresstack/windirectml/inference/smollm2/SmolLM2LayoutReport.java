package com.aresstack.windirectml.inference.smollm2;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Diagnostic result for SmolLM2 tensor layout analysis.
 */
public record SmolLM2LayoutReport(
        String modelFamily,
        boolean layoutComplete,
        boolean runtimeLoadable,
        String runtimeLoadableReason,
        int foundTensorCount,
        int knownTensorCount,
        int unknownTensorCount,
        List<String> missingRequiredRoles,
        List<String> shapeErrors,
        List<Integer> detectedLayers,
        Set<String> detectedDTypes,
        boolean usesGroupedQueryAttention,
        boolean usesTiedEmbeddings,
        List<SmolLM2TensorRoleBinding> roles
) {
    public static final String RUNTIME_NOT_IMPLEMENTED = "SmolLM2 runtime is not implemented yet";

    public SmolLM2LayoutReport {
        missingRequiredRoles = List.copyOf(missingRequiredRoles);
        shapeErrors = List.copyOf(shapeErrors);
        detectedLayers = List.copyOf(detectedLayers);
        detectedDTypes = Set.copyOf(detectedDTypes);
        roles = List.copyOf(roles);
    }

    public Map<String, Object> toManifest() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("modelFamily", modelFamily);
        out.put("layoutComplete", layoutComplete);
        out.put("runtimeLoadable", runtimeLoadable);
        out.put("runtimeLoadableReason", runtimeLoadableReason);
        out.put("foundTensorCount", foundTensorCount);
        out.put("knownTensorCount", knownTensorCount);
        out.put("unknownTensorCount", unknownTensorCount);
        out.put("missingRequiredRoles", missingRequiredRoles);
        out.put("shapeErrors", shapeErrors);
        out.put("detectedLayers", detectedLayers);
        out.put("detectedDTypes", detectedDTypes.stream().sorted().toList());
        out.put("usesGroupedQueryAttention", usesGroupedQueryAttention);
        out.put("usesTiedEmbeddings", usesTiedEmbeddings);
        out.put("roles", roles.stream().map(SmolLM2TensorRoleBinding::toManifest).toList());
        return out;
    }
}
