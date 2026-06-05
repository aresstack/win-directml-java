package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.model.RuntimeTensor;
import com.aresstack.windirectml.inference.model.RuntimeTensorCatalog;

import java.io.IOException;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Resolves wdmlpack runtime tensors into the SmolLM2 weight model.
 */
public final class SmolLM2WeightResolver {

    private final SmolLM2TensorNameMapper mapper;
    private final SmolLM2LayoutValidator layoutValidator;

    public SmolLM2WeightResolver() {
        this(new SmolLM2TensorNameMapper(), new SmolLM2LayoutValidator());
    }

    SmolLM2WeightResolver(SmolLM2TensorNameMapper mapper, SmolLM2LayoutValidator layoutValidator) {
        this.mapper = Objects.requireNonNull(mapper, "mapper");
        this.layoutValidator = Objects.requireNonNull(layoutValidator, "layoutValidator");
    }

    public SmolLM2Weights resolve(SmolLM2Config config, RuntimeTensorCatalog catalog) throws IOException {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(catalog, "catalog");
        if (catalog.isEmpty()) {
            throw new IOException("SmolLM2 wdmlpack does not contain tensor payloads");
        }
        SmolLM2LayoutReport report = layoutValidator.validate(config, catalog.toSourceTensorCatalog());
        if (!report.layoutComplete()) {
            throw new IOException("SmolLM2 wdmlpack tensor layout is incomplete: missing="
                    + report.missingRequiredRoles() + ", shapeErrors=" + report.shapeErrors());
        }
        Map<String, SmolLM2WeightTensor> byRoleKey = bindRuntimeTensors(catalog);
        SmolLM2WeightTensor tokenEmbedding = require(byRoleKey, SmolLM2TensorRole.TOKEN_EMBEDDING.name());
        SmolLM2WeightTensor finalNorm = require(byRoleKey, SmolLM2TensorRole.FINAL_NORM.name());
        SmolLM2WeightTensor lmHead = byRoleKey.get(SmolLM2TensorRole.LM_HEAD.name());
        boolean lmHeadTiedToEmbedding = false;
        if (lmHead == null) {
            if (!config.tieWordEmbeddings()) {
                throw new IOException("SmolLM2 wdmlpack is missing LM_HEAD and tie_word_embeddings=false");
            }
            lmHead = tokenEmbedding;
            lmHeadTiedToEmbedding = true;
        }
        List<SmolLM2LayerWeights> layers = resolveLayers(config, byRoleKey);
        return new SmolLM2Weights(config, tokenEmbedding, finalNorm, lmHead, lmHeadTiedToEmbedding,
                layers, catalog.payloadBytes());
    }

    private Map<String, SmolLM2WeightTensor> bindRuntimeTensors(RuntimeTensorCatalog catalog) throws IOException {
        Map<String, SmolLM2WeightTensor> byRoleKey = new LinkedHashMap<>();
        for (RuntimeTensor tensor : catalog.values()) {
            var mapped = mapper.map(tensor.name());
            if (mapped.isEmpty()) {
                continue;
            }
            SmolLM2TensorRoleBinding binding = mapped.get();
            SmolLM2WeightTensor previous = byRoleKey.putIfAbsent(binding.key(), new SmolLM2WeightTensor(binding, tensor));
            if (previous != null) {
                throw new IOException("Duplicate SmolLM2 runtime tensor role " + binding.key()
                        + " from " + previous.tensorName() + " and " + tensor.name());
            }
        }
        return byRoleKey;
    }

    private static List<SmolLM2LayerWeights> resolveLayers(SmolLM2Config config,
                                                           Map<String, SmolLM2WeightTensor> byRoleKey)
            throws IOException {
        List<SmolLM2LayerWeights> layers = new ArrayList<>();
        for (int layer = 0; layer < config.numHiddenLayers(); layer++) {
            EnumMap<SmolLM2TensorRole, SmolLM2WeightTensor> tensors = new EnumMap<>(SmolLM2TensorRole.class);
            for (SmolLM2TensorRole role : layerRoles()) {
                tensors.put(role, require(byRoleKey, role.name() + "#" + layer));
            }
            layers.add(new SmolLM2LayerWeights(layer, tensors));
        }
        return layers;
    }

    private static List<SmolLM2TensorRole> layerRoles() {
        return List.of(
                SmolLM2TensorRole.LAYER_INPUT_NORM,
                SmolLM2TensorRole.LAYER_SELF_Q,
                SmolLM2TensorRole.LAYER_SELF_K,
                SmolLM2TensorRole.LAYER_SELF_V,
                SmolLM2TensorRole.LAYER_SELF_O,
                SmolLM2TensorRole.LAYER_POST_ATTENTION_NORM,
                SmolLM2TensorRole.LAYER_MLP_GATE,
                SmolLM2TensorRole.LAYER_MLP_UP,
                SmolLM2TensorRole.LAYER_MLP_DOWN);
    }

    private static SmolLM2WeightTensor require(Map<String, SmolLM2WeightTensor> byRoleKey, String key)
            throws IOException {
        SmolLM2WeightTensor tensor = byRoleKey.get(key);
        if (tensor == null) {
            throw new IOException("Missing SmolLM2 runtime tensor role " + key);
        }
        return tensor;
    }
}
