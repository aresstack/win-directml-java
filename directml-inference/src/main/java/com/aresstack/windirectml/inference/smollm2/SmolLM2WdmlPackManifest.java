package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.model.SourceTensor;
import com.aresstack.windirectml.inference.model.SourceTensorCatalog;
import com.aresstack.windirectml.inference.model.WdmlPackWriter;

import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Builds the SmolLM2 wdmlpack manifest contract.
 */
public final class SmolLM2WdmlPackManifest {

    public static final int SCHEMA_VERSION = WdmlPackWriter.VERSION;
    public static final int COMPILER_VERSION = 42;

    private SmolLM2WdmlPackManifest() {
    }

    public static Map<String, Object> build(SmolLM2Config config,
                                            SmolLM2LayoutReport layoutReport,
                                            SmolLM2PayloadPolicy payloadPolicy) {
        return build(config, layoutReport, payloadPolicy, null, List.of(), null);
    }

    public static Map<String, Object> build(SmolLM2Config config,
                                            SmolLM2LayoutReport layoutReport,
                                            SmolLM2PayloadPolicy payloadPolicy,
                                            SourceTensorCatalog catalog,
                                            List<TensorPayloadPlan> payloadPlan) {
        return build(config, layoutReport, payloadPolicy, catalog, payloadPlan, null);
    }

    public static Map<String, Object> build(SmolLM2Config config,
                                            SmolLM2LayoutReport layoutReport,
                                            SmolLM2PayloadPolicy payloadPolicy,
                                            SourceTensorCatalog catalog,
                                            List<TensorPayloadPlan> payloadPlan,
                                            SmolLM2ModelDirectory.SourceAggregate source) {
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("format", "wdmlpack");
        root.put("version", SCHEMA_VERSION);
        root.put("schemaVersion", SCHEMA_VERSION);
        root.put("compilerVersion", COMPILER_VERSION);
        root.put("createdAt", Instant.now().toString());
        root.put("modelFamily", "smollm2");
        root.put("architecture", "llama-causal-decoder");
        SmolLM2RuntimeLoadability loadability = SmolLM2RuntimeLoadability.forPackage(
                layoutReport, payloadPolicy.payloadIncluded());
        root.put("sourceFormat", "safetensors");
        root.put("payloadIncluded", payloadPolicy.payloadIncluded());
        root.put("weightsLoadable", payloadPolicy.payloadIncluded() && layoutReport.layoutComplete());
        root.put("runtimeLoadable", loadability.runtimeLoadable());
        root.put("runtimeLoadMode", loadability.runtimeLoadMode());
        root.put("runtimeLoadableReason", loadability.runtimeLoadableReason());
        root.put("layoutComplete", layoutReport.layoutComplete());
        if (source != null) {
            root.put("source", source.toManifest());
        }
        root.put("model", SmolLM2PackageMetadata.from(config).toManifest());
        root.put("smollm2Layout", layoutReport.toManifest());
        root.put("tensorCatalog", tensorCatalog(layoutReport, catalog));
        root.put("tensors", tensorEntries(catalog, payloadPlan));
        return root;
    }

    public record TensorPayloadPlan(String name, long payloadOffset, long payloadLength) {
    }

    private static Map<String, Object> tensorCatalog(SmolLM2LayoutReport layoutReport, SourceTensorCatalog catalog) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("count", layoutReport.foundTensorCount());
        out.put("knownTensorCount", layoutReport.knownTensorCount());
        out.put("unknownTensorCount", layoutReport.unknownTensorCount());
        if (catalog != null) {
            out.put("inlineBytes", catalog.inlineBytes());
            out.put("externalBytes", catalog.externalBytes());
        }
        return out;
    }

    private static List<Map<String, Object>> tensorEntries(SourceTensorCatalog catalog,
                                                           List<TensorPayloadPlan> payloadPlan) {
        if (catalog == null) {
            return List.of();
        }
        Map<String, TensorPayloadPlan> planByName = new LinkedHashMap<>();
        for (TensorPayloadPlan plan : payloadPlan) {
            planByName.put(plan.name(), plan);
        }
        List<Map<String, Object>> entries = new ArrayList<>();
        for (SourceTensor tensor : catalog.entries().values()) {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("name", tensor.name());
            out.put("dataType", tensor.onnxDataType());
            out.put("dataTypeName", tensor.dataTypeName());
            out.put("dims", tensor.dims());
            out.put("byteLength", tensor.byteLength());
            TensorPayloadPlan plan = planByName.get(tensor.name());
            if (plan != null) {
                out.put("payloadOffset", plan.payloadOffset());
                out.put("payloadLength", plan.payloadLength());
            }
            entries.add(out);
        }
        return entries;
    }
}
