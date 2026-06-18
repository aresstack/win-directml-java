package com.aresstack.windirectml.inference.phi3;

import com.aresstack.windirectml.inference.model.RuntimeModelPackage;
import com.aresstack.windirectml.inference.model.RuntimeTensor;
import com.aresstack.windirectml.inference.model.RuntimeTensorCatalog;
import com.aresstack.windirectml.inference.model.WdmlPackManifest;
import com.aresstack.windirectml.inference.phi3.Phi3Weights.LayerWeights;
import com.aresstack.windirectml.inference.phi3.Phi3Weights.QuantizedWeight;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;

/**
 * Package-backed loader for a Phi-3 {@code model_phi3.wdmlpack} (PHI3-WDMLPACK-COMPILER-1).
 *
 * <p>Opens the container via the shared {@link RuntimeModelPackage}/{@code WdmlPackReader}, validates it is a Phi-3
 * payload package, and reconstructs the {@link Phi3Config} and {@link Phi3Weights} records so {@link Phi3Runtime} can
 * be driven unchanged from a compiled package instead of raw ONNX. This is the runtime seam named in the audit; the
 * runtime math itself is untouched.</p>
 */
public final class Phi3RuntimePackage {

    private final RuntimeModelPackage modelPackage;
    private final Phi3Config config;

    private Phi3RuntimePackage(RuntimeModelPackage modelPackage, Phi3Config config) {
        this.modelPackage = modelPackage;
        this.config = config;
    }

    public static Phi3RuntimePackage open(Path packagePath) throws IOException {
        RuntimeModelPackage pkg = RuntimeModelPackage.open(packagePath);
        String architecture = WdmlPackManifest.stringValue(pkg.manifest().get("architecture"));
        if (!Phi3WdmlPackCompiler.ARCHITECTURE.equals(architecture)) {
            throw new IOException("Not a Phi-3 wdmlpack (architecture=" + architecture + "): " + packagePath);
        }
        if (!pkg.payloadIncluded() || !pkg.runtimeLoadable()) {
            throw new IOException("Phi-3 wdmlpack is not a runtime-loadable payload package: " + packagePath);
        }
        return new Phi3RuntimePackage(pkg, readConfig(pkg));
    }

    public Phi3Config config() {
        return config;
    }

    public Path packagePath() {
        return modelPackage.packagePath();
    }

    public String runtimeLoadMode() {
        return modelPackage.runtimeLoadMode();
    }

    /** Reconstruct the in-memory weights the Phi-3 runtime consumes. */
    public Phi3Weights weights() throws IOException {
        RuntimeTensorCatalog catalog = modelPackage.runtimeTensorCatalog();

        float[] embed = floats(catalog, Phi3WdmlPackRoles.EMBED_TOKENS);
        float[] cos = floats(catalog, Phi3WdmlPackRoles.COS_CACHE);
        float[] sin = floats(catalog, Phi3WdmlPackRoles.SIN_CACHE);
        float[] finalNorm = floats(catalog, Phi3WdmlPackRoles.FINAL_NORM);
        QuantizedWeight lmHead = quantized(catalog, Phi3WdmlPackRoles.LM_HEAD);

        LayerWeights[] layers = new LayerWeights[config.numHiddenLayers()];
        for (int l = 0; l < layers.length; l++) {
            layers[l] = new LayerWeights(
                    floats(catalog, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.INPUT_NORM)),
                    quantized(catalog, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.Q_PROJ)),
                    quantized(catalog, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.K_PROJ)),
                    quantized(catalog, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.V_PROJ)),
                    floats(catalog, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.ATTN_OUT_SCALE)),
                    quantized(catalog, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.O_PROJ)),
                    floats(catalog, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.POST_NORM)),
                    quantized(catalog, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.GATE_UP_PROJ)),
                    floats(catalog, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.MLP_OUT_SCALE)),
                    quantized(catalog, Phi3WdmlPackRoles.layer(l, Phi3WdmlPackRoles.DOWN_PROJ)));
        }
        return Phi3Weights.ofRecords(config, embed, cos, sin, layers, finalNorm, lmHead);
    }

    private static Phi3Config readConfig(RuntimeModelPackage pkg) throws IOException {
        Map<String, Object> model = pkg.requireMap("model");
        return new Phi3Config(
                WdmlPackManifest.intValue(model.get("hiddenSize"), 0),
                WdmlPackManifest.intValue(model.get("numAttentionHeads"), 0),
                WdmlPackManifest.intValue(model.get("numHiddenLayers"), 0),
                WdmlPackManifest.intValue(model.get("numKeyValueHeads"), 0),
                WdmlPackManifest.intValue(model.get("vocabSize"), 0),
                WdmlPackManifest.intValue(model.get("maxPositionEmbeddings"), 0),
                WdmlPackManifest.intValue(model.get("intermediateSize"), 0),
                (float) doubleValue(model.get("rmsNormEps")),
                (float) doubleValue(model.get("ropeTheta")));
    }

    private static double doubleValue(Object value) {
        if (value instanceof Number n) {
            return n.doubleValue();
        }
        if (value instanceof String s && !s.isBlank()) {
            try {
                return Double.parseDouble(s);
            } catch (NumberFormatException ignored) {
                return 0.0;
            }
        }
        return 0.0;
    }

    private static float[] floats(RuntimeTensorCatalog catalog, String name) throws IOException {
        RuntimeTensor tensor = require(catalog, name);
        ByteBuffer buf = tensor.rawDataBuffer().order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[tensor.rawByteLength() / Float.BYTES];
        for (int i = 0; i < out.length; i++) {
            out[i] = buf.getFloat();
        }
        return out;
    }

    private static byte[] bytes(RuntimeTensorCatalog catalog, String name) throws IOException {
        RuntimeTensor tensor = require(catalog, name);
        ByteBuffer buf = tensor.rawDataBuffer();
        byte[] out = new byte[tensor.rawByteLength()];
        buf.get(out);
        return out;
    }

    private static QuantizedWeight quantized(RuntimeTensorCatalog catalog, String role) throws IOException {
        RuntimeTensor q = require(catalog, Phi3WdmlPackRoles.qweight(role));
        long[] dims = q.dims();
        if (dims.length != 3) {
            throw new IOException("Phi-3 qweight '" + role + "' must encode dims [N,K,blockSize], got "
                    + dims.length + " dims");
        }
        byte[] qWeight = bytes(catalog, Phi3WdmlPackRoles.qweight(role));
        float[] scales = floats(catalog, Phi3WdmlPackRoles.scales(role));
        byte[] zeroPoints = bytes(catalog, Phi3WdmlPackRoles.zeropoints(role));
        return new QuantizedWeight(qWeight, scales, zeroPoints,
                (int) dims[0], (int) dims[1], (int) dims[2]);
    }

    private static RuntimeTensor require(RuntimeTensorCatalog catalog, String name) throws IOException {
        RuntimeTensor tensor = catalog.get(name);
        if (tensor == null || !tensor.hasPayload()) {
            throw new IOException("Phi-3 wdmlpack is missing payload tensor: " + name);
        }
        return tensor;
    }

    @Override
    public String toString() {
        return "Phi3RuntimePackage{" + Objects.toString(modelPackage) + "}";
    }
}
