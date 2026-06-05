package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.model.SourceTensor;
import com.aresstack.windirectml.inference.model.SourceTensorCatalog;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Validates SmolLM2 SafeTensors names and shapes against config.json.
 */
public final class SmolLM2LayoutValidator {

    private final SmolLM2TensorNameMapper mapper;

    public SmolLM2LayoutValidator() {
        this(new SmolLM2TensorNameMapper());
    }

    public SmolLM2LayoutValidator(SmolLM2TensorNameMapper mapper) {
        this.mapper = mapper;
    }

    public SmolLM2LayoutReport validate(SmolLM2Config config, SourceTensorCatalog catalog) {
        SmolLM2Architecture architecture = SmolLM2Architecture.from(config);
        Map<String, SmolLM2TensorRoleBinding> rolesByKey = new LinkedHashMap<>();
        List<SmolLM2TensorRoleBinding> roles = new ArrayList<>();
        List<String> missing = new ArrayList<>();
        List<String> shapeErrors = new ArrayList<>();
        Set<Integer> layers = new LinkedHashSet<>();
        Set<String> dtypes = new LinkedHashSet<>();
        int known = 0;
        int unknown = 0;

        for (SourceTensor tensor : catalog.entries().values()) {
            if (tensor.dataTypeName() != null && !tensor.dataTypeName().isBlank()) {
                dtypes.add(tensor.dataTypeName());
            }
            var mapped = mapper.map(tensor.name());
            if (mapped.isEmpty()) {
                unknown++;
                continue;
            }
            known++;
            SmolLM2TensorRoleBinding binding = mapped.get();
            if (binding.role().layerBound()) {
                layers.add(binding.layerIndex());
            }
            SmolLM2TensorRoleBinding previous = rolesByKey.putIfAbsent(binding.key(), binding);
            if (previous != null) {
                shapeErrors.add("duplicate role " + binding.key() + " from " + previous.tensorName()
                        + " and " + binding.tensorName());
                continue;
            }
            roles.add(binding);
            assertShape(config, architecture, tensor, binding, shapeErrors);
        }

        requireRootRole(rolesByKey, SmolLM2TensorRole.TOKEN_EMBEDDING, missing);
        requireRootRole(rolesByKey, SmolLM2TensorRole.FINAL_NORM, missing);
        if (!config.tieWordEmbeddings()) {
            requireRootRole(rolesByKey, SmolLM2TensorRole.LM_HEAD, missing);
        }
        requireLayers(config, rolesByKey, missing);

        List<Integer> detectedLayers = layers.stream().sorted().toList();
        boolean complete = missing.isEmpty() && shapeErrors.isEmpty();
        roles.sort(Comparator.comparing(SmolLM2TensorRoleBinding::key));
        return new SmolLM2LayoutReport(
                "smollm2",
                complete,
                complete,
                complete ? SmolLM2LayoutReport.REFERENCE_RUNTIME_AVAILABLE : SmolLM2LayoutReport.INCOMPLETE_LAYOUT,
                catalog.size(),
                known,
                unknown,
                missing,
                shapeErrors,
                detectedLayers,
                dtypes,
                architecture.usesGroupedQueryAttention(),
                architecture.usesTiedWordEmbeddings(),
                roles);
    }

    private static void requireRootRole(Map<String, SmolLM2TensorRoleBinding> rolesByKey,
                                        SmolLM2TensorRole role,
                                        List<String> missing) {
        if (!rolesByKey.containsKey(role.name())) {
            missing.add(role.name());
        }
    }

    private static void requireLayers(SmolLM2Config config,
                                      Map<String, SmolLM2TensorRoleBinding> rolesByKey,
                                      List<String> missing) {
        List<SmolLM2TensorRole> required = List.of(
                SmolLM2TensorRole.LAYER_INPUT_NORM,
                SmolLM2TensorRole.LAYER_SELF_Q,
                SmolLM2TensorRole.LAYER_SELF_K,
                SmolLM2TensorRole.LAYER_SELF_V,
                SmolLM2TensorRole.LAYER_SELF_O,
                SmolLM2TensorRole.LAYER_POST_ATTENTION_NORM,
                SmolLM2TensorRole.LAYER_MLP_GATE,
                SmolLM2TensorRole.LAYER_MLP_UP,
                SmolLM2TensorRole.LAYER_MLP_DOWN);
        for (int layer = 0; layer < config.numHiddenLayers(); layer++) {
            for (SmolLM2TensorRole role : required) {
                String key = role.name() + "#" + layer;
                if (!rolesByKey.containsKey(key)) {
                    missing.add(key);
                }
            }
        }
    }

    private static void assertShape(SmolLM2Config config,
                                    SmolLM2Architecture architecture,
                                    SourceTensor tensor,
                                    SmolLM2TensorRoleBinding binding,
                                    List<String> shapeErrors) {
        long[] expected = expectedShape(config, architecture, binding.role());
        long[] actual = tensor.dims();
        if (!sameShape(expected, actual)) {
            shapeErrors.add(binding.key() + " shape mismatch for " + tensor.name()
                    + ": expected " + shape(expected) + ", got " + shape(actual));
        }
    }

    private static long[] expectedShape(SmolLM2Config config,
                                        SmolLM2Architecture architecture,
                                        SmolLM2TensorRole role) {
        int hidden = config.hiddenSize();
        int intermediate = config.intermediateSize();
        int qRows = architecture.attentionHeads() * architecture.headDim();
        int kvRows = architecture.keyValueHeads() * architecture.headDim();
        return switch (role) {
            case TOKEN_EMBEDDING, LM_HEAD -> new long[]{config.vocabSize(), hidden};
            case FINAL_NORM, LAYER_INPUT_NORM, LAYER_POST_ATTENTION_NORM -> new long[]{hidden};
            case LAYER_SELF_Q -> new long[]{qRows, hidden};
            case LAYER_SELF_K, LAYER_SELF_V -> new long[]{kvRows, hidden};
            case LAYER_SELF_O -> new long[]{hidden, qRows};
            case LAYER_MLP_GATE, LAYER_MLP_UP -> new long[]{intermediate, hidden};
            case LAYER_MLP_DOWN -> new long[]{hidden, intermediate};
        };
    }

    private static boolean sameShape(long[] expected, long[] actual) {
        if (expected.length != actual.length) {
            return false;
        }
        for (int i = 0; i < expected.length; i++) {
            if (expected[i] != actual[i]) {
                return false;
            }
        }
        return true;
    }

    private static String shape(long[] dims) {
        StringBuilder builder = new StringBuilder("[");
        for (int i = 0; i < dims.length; i++) {
            if (i > 0) {
                builder.append(", ");
            }
            builder.append(dims[i]);
        }
        return builder.append(']').toString();
    }
}
