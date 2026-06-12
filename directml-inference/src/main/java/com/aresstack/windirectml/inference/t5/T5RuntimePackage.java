package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.model.RuntimeLoadability;
import com.aresstack.windirectml.inference.model.RuntimeModelPackage;
import com.aresstack.windirectml.inference.model.RuntimeTensor;
import com.aresstack.windirectml.inference.model.RuntimeTensorCatalog;
import com.aresstack.windirectml.inference.model.WdmlPackWriter;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * T5-local view of an internal {@code .wdmlpack} runtime package.
 *
 * <p>The package reader is the first real T5 runtime boundary: it validates the
 * model-family manifest, maps the package payload, resolves normalized T5 roles,
 * and exposes {@link T5Weights}. It deliberately does not execute the model yet;
 * encoder/decoder execution is introduced by later runtime patches.</p>
 */
public final class T5RuntimePackage {
    private final Path packagePath;
    private final RuntimeModelPackage modelPackage;
    private final Map<String, Object> manifest;
    private final T5PackageMetadata metadata;
    private final RuntimeTensorCatalog tensorCatalog;
    private final Map<String, T5TensorRole> rolesByRole;
    private final Map<String, T5TensorRole> rolesByRuntimeName;
    private final T5Weights weights;

    private T5RuntimePackage(Path packagePath,
                             RuntimeModelPackage modelPackage,
                             Map<String, Object> manifest,
                             T5PackageMetadata metadata,
                             RuntimeTensorCatalog tensorCatalog,
                             Collection<T5TensorRole> roles,
                             boolean loadWeights) throws IOException {
        this.packagePath = packagePath;
        this.modelPackage = modelPackage;
        this.manifest = Map.copyOf(new LinkedHashMap<>(manifest));
        this.metadata = Objects.requireNonNull(metadata, "metadata");
        this.tensorCatalog = Objects.requireNonNull(tensorCatalog, "tensorCatalog");
        this.rolesByRole = indexByRole(roles);
        this.rolesByRuntimeName = indexByRuntimeName(roles);
        this.weights = loadWeights ? T5Weights.load(this) : null;
    }

    public static T5RuntimePackage open(Path packagePath) throws IOException {
        RuntimeModelPackage modelPackage = RuntimeModelPackage.open(packagePath);
        Map<String, Object> manifest = modelPackage.manifest();
        validateT5Manifest(manifest);
        T5PackageMetadata metadata = metadataFromManifest(manifest);
        RuntimeTensorCatalog tensorCatalog = modelPackage.payloadIncluded()
                ? modelPackage.runtimeTensorCatalog()
                : RuntimeTensorCatalog.empty();
        List<T5TensorRole> roles = rolesFromManifest(manifest);
        boolean loadWeights = Boolean.TRUE.equals(manifest.get("weightsLoadable"));
        return new T5RuntimePackage(modelPackage.packagePath(), modelPackage, manifest, metadata, tensorCatalog, roles, loadWeights);
    }

    public static T5RuntimePackage fromMetadata(T5PackageMetadata metadata) {
        try {
            Map<String, Object> manifest = new LinkedHashMap<>();
            manifest.put("format", "wdmlpack");
            manifest.put("version", WdmlPackWriter.VERSION);
            manifest.put("modelFamily", T5PackageMetadata.MODEL_FAMILY);
            manifest.put("architecture", T5PackageMetadata.ARCHITECTURE);
            manifest.put("payloadIncluded", false);
            manifest.put("weightsLoadable", false);
            manifest.put("runtimeLoadable", false);
            manifest.put("runtimeLoadMode", T5ManifestPayloadPolicy.RUNTIME_LOAD_MODE);
            manifest.put("reason", T5ManifestPayloadPolicy.REASON);
            manifest.put("t5", metadata.toManifest());
            return new T5RuntimePackage(null, null, manifest, metadata, RuntimeTensorCatalog.empty(), List.of(), false);
        } catch (IOException e) {
            throw new IllegalStateException("Could not create metadata-only T5 runtime package", e);
        }
    }

    public Path packagePath() {
        return packagePath;
    }

    public RuntimeModelPackage modelPackage() {
        return modelPackage;
    }

    public Map<String, Object> manifest() {
        return manifest;
    }

    public T5PackageMetadata metadata() {
        return metadata;
    }

    public RuntimeTensorCatalog tensorCatalog() {
        return tensorCatalog;
    }

    public boolean payloadIncluded() {
        return Boolean.TRUE.equals(manifest.get("payloadIncluded"));
    }

    public boolean weightsLoadable() {
        return Boolean.TRUE.equals(manifest.get("weightsLoadable"));
    }

    public boolean runtimeLoadable() {
        return Boolean.TRUE.equals(manifest.get("runtimeLoadable"));
    }

    /**
     * Family-neutral {@link RuntimeLoadability} view, derived faithfully from this package's manifest fields
     * ({@code runtimeLoadable}/{@code runtimeLoadMode}/{@code reason}). This is the same shared report shape the
     * decoder-only family uses, so callers can read loadability uniformly. (Note: the T5 manifest's
     * {@code runtimeLoadable} flag is a pre-existing manifest value and is independent of this view.)
     */
    public RuntimeLoadability loadability() {
        String mode = stringValue(manifest.get("runtimeLoadMode"));
        String reason = stringValue(manifest.get("reason"));
        return new RuntimeLoadability(
                runtimeLoadable(),
                mode.isBlank() ? "unknown" : mode,
                reason.isBlank() ? (runtimeLoadable() ? "runtime-loadable" : "runtime-not-loadable") : reason);
    }

    public T5Weights weights() throws IOException {
        if (weights == null) {
            throw new IOException("T5 package does not contain loadable weights: " + packagePath);
        }
        return weights;
    }

    public Collection<T5TensorRole> roles() {
        return rolesByRole.values();
    }

    public T5TensorRole role(String role) {
        return rolesByRole.get(role);
    }

    RuntimeTensor tensorForRuntimeName(String runtimeName) {
        return tensorCatalog.get(runtimeName);
    }

    static void validateT5Manifest(Map<String, Object> manifest) throws IOException {
        String modelFamily = stringValue(manifest.get("modelFamily"));
        if (modelFamily.isBlank() && manifest.get("t5") instanceof Map<?, ?> t5) {
            modelFamily = stringValue(t5.get("modelFamily"));
        }
        if (!T5PackageMetadata.MODEL_FAMILY.equals(modelFamily)) {
            throw new IOException("Not a T5 wdmlpack manifest: modelFamily=" + modelFamily);
        }
        String architecture = stringValue(manifest.get("architecture"));
        if (!architecture.isBlank() && !T5PackageMetadata.ARCHITECTURE.equals(architecture)) {
            throw new IOException("Not a T5 encoder-decoder wdmlpack manifest: architecture=" + architecture);
        }
    }

    @SuppressWarnings("unchecked")
    private static T5PackageMetadata metadataFromManifest(Map<String, Object> manifest) throws IOException {
        Object t5 = manifest.get("t5");
        if (!(t5 instanceof Map<?, ?> raw)) {
            throw new IOException("Invalid T5 wdmlpack manifest: missing t5 section");
        }
        Map<String, Object> map = (Map<String, Object>) raw;
        T5Config config = new T5Config(java.util.List.of("T5ForConditionalGeneration"), "t5", true,
                intValue(map.get("dModel")), intValue(map.get("dKv")), intValue(map.get("dFf")),
                intValue(map.get("encoderLayers")), intValue(map.get("decoderLayers")), intValue(map.get("numHeads")),
                intValue(map.get("vocabSize")), intValue(map.get("relativeAttentionBuckets")),
                intValue(map.get("relativeAttentionMaxDistance")), 0.0f,
                intValue(map.get("decoderStartTokenId")), intValue(map.get("eosTokenId")), intValue(map.get("padTokenId")),
                Boolean.valueOf(String.valueOf(map.getOrDefault("tieWordEmbeddings", Boolean.TRUE))),
                stringValue(map.get("feedForwardProjection")));
        try {
            config.validate();
        } catch (IOException e) {
            throw new IOException("Invalid T5 package metadata: " + e.getMessage(), e);
        }
        return T5PackageMetadata.from(config);
    }

    @SuppressWarnings("unchecked")
    private static List<T5TensorRole> rolesFromManifest(Map<String, Object> manifest) throws IOException {
        Object tensors = manifest.get("tensors");
        if (!(tensors instanceof List<?> tensorList)) {
            Object layout = manifest.get("layout");
            if (layout instanceof Map<?, ?> rawLayout) {
                tensors = ((Map<String, Object>) rawLayout).get("roles");
            }
        }
        if (!(tensors instanceof List<?> tensorList)) {
            return List.of();
        }
        java.util.ArrayList<T5TensorRole> roles = new java.util.ArrayList<>();
        for (Object item : tensorList) {
            if (!(item instanceof Map<?, ?> raw)) {
                continue;
            }
            Map<String, Object> map = (Map<String, Object>) raw;
            String role = stringValue(map.get("role"));
            String sourceName = stringValue(map.get("sourceName"));
            String runtimeName = stringValue(map.get("runtimeName"));
            if (role.isBlank() || runtimeName.isBlank()) {
                continue;
            }
            roles.add(new T5TensorRole(role, sourceName, runtimeName,
                    intValue(map.get("dataType")), stringValue(map.get("dataTypeName")),
                    RuntimeModelPackage.dimsValue(map.get("dims")),
                    booleanValue(map.get("required")), booleanValue(map.get("tied"))));
        }
        return List.copyOf(roles);
    }

    private static Map<String, T5TensorRole> indexByRole(Collection<T5TensorRole> roles) throws IOException {
        LinkedHashMap<String, T5TensorRole> out = new LinkedHashMap<>();
        for (T5TensorRole role : roles) {
            T5TensorRole previous = out.put(role.role(), role);
            if (previous != null) {
                throw new IOException("Duplicate T5 tensor role in package manifest: " + role.role());
            }
        }
        return Map.copyOf(out);
    }

    private static Map<String, T5TensorRole> indexByRuntimeName(Collection<T5TensorRole> roles) throws IOException {
        LinkedHashMap<String, T5TensorRole> out = new LinkedHashMap<>();
        for (T5TensorRole role : roles) {
            if (role.tied()) {
                continue;
            }
            T5TensorRole previous = out.put(role.runtimeName(), role);
            if (previous != null) {
                throw new IOException("Duplicate T5 runtime tensor name in package manifest: " + role.runtimeName());
            }
        }
        return Map.copyOf(out);
    }

    private static int intValue(Object value) {
        if (value instanceof Number n) {
            return n.intValue();
        }
        if (value instanceof String s && !s.isBlank()) {
            return Integer.parseInt(s);
        }
        return 0;
    }

    private static boolean booleanValue(Object value) {
        if (value instanceof Boolean b) {
            return b;
        }
        if (value instanceof String s) {
            return Boolean.parseBoolean(s);
        }
        return false;
    }

    private static String stringValue(Object value) {
        return value instanceof String s ? s : "";
    }
}
