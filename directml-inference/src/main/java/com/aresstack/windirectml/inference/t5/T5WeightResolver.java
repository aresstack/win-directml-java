package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.model.RuntimeTensor;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Resolves normalized T5 manifest roles to payload-backed runtime tensors.
 */
final class T5WeightResolver {
    private final T5RuntimePackage runtimePackage;

    T5WeightResolver(T5RuntimePackage runtimePackage) {
        this.runtimePackage = Objects.requireNonNull(runtimePackage, "runtimePackage");
    }

    Map<String, RuntimeTensor> resolve() throws IOException {
        if (!runtimePackage.payloadIncluded()) {
            throw new IOException("T5 wdmlpack does not include tensor payloads: " + runtimePackage.packagePath());
        }
        Map<String, RuntimeTensor> resolved = new LinkedHashMap<>();
        for (T5TensorRole role : runtimePackage.roles()) {
            RuntimeTensor tensor = resolve(role, resolved);
            if (tensor != null) {
                resolved.put(role.role(), tensor);
            }
        }
        return Map.copyOf(resolved);
    }

    private RuntimeTensor resolve(T5TensorRole role, Map<String, RuntimeTensor> alreadyResolved) throws IOException {
        if (role.tied()) {
            return resolveTied(role, alreadyResolved);
        }
        RuntimeTensor tensor = runtimePackage.tensorForRuntimeName(role.runtimeName());
        if (tensor == null || !tensor.hasPayload()) {
            if (role.required()) {
                throw new IOException("Missing payload for required T5 role " + role.role()
                        + " (runtimeName=" + role.runtimeName() + ")");
            }
            return null;
        }
        validateShape(role, tensor);
        validateDataType(role, tensor);
        return tensor;
    }

    private RuntimeTensor resolveTied(T5TensorRole role, Map<String, RuntimeTensor> alreadyResolved) throws IOException {
        if ("lm_head".equals(role.role())) {
            RuntimeTensor shared = alreadyResolved.get("shared_embedding");
            if (shared != null) {
                return shared;
            }
            T5TensorRole sharedRole = runtimePackage.role("shared_embedding");
            if (sharedRole == null) {
                throw new IOException("T5 tied lm_head requires shared_embedding role");
            }
            RuntimeTensor tensor = runtimePackage.tensorForRuntimeName(sharedRole.runtimeName());
            if (tensor == null || !tensor.hasPayload()) {
                throw new IOException("T5 tied lm_head requires shared embedding payload");
            }
            return tensor;
        }
        throw new IOException("Unsupported tied T5 role: " + role.role());
    }

    private static void validateShape(T5TensorRole role, RuntimeTensor tensor) throws IOException {
        long[] expected = role.dims();
        long[] actual = tensor.dims();
        if (expected.length != actual.length) {
            throw shapeError(role, actual);
        }
        for (int i = 0; i < expected.length; i++) {
            if (expected[i] != actual[i]) {
                throw shapeError(role, actual);
            }
        }
    }

    private static IOException shapeError(T5TensorRole role, long[] actual) {
        return new IOException("T5 tensor shape mismatch for role " + role.role()
                + ": expected=" + java.util.Arrays.toString(role.dims())
                + ", actual=" + java.util.Arrays.toString(actual));
    }

    private static void validateDataType(T5TensorRole role, RuntimeTensor tensor) throws IOException {
        if (role.dataType() != 0 && role.dataType() != tensor.dataType()) {
            throw new IOException("T5 tensor dtype mismatch for role " + role.role()
                    + ": expected=" + role.dataType() + ", actual=" + tensor.dataType());
        }
    }
}
