package com.aresstack.windirectml.inference.piper;

import com.aresstack.windirectml.windows.OnnxModelReader.OnnxNode;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Execute a bounded prefix of the real Piper VITS graph using native Java reference kernels.
 * Keep this diagnostic interpreter separate from the future optimized DirectML graph executor.
 */
public final class PiperNativeGraphPrefixRunner {

    private final long randomSeed;

    public PiperNativeGraphPrefixRunner() {
        this(42L);
    }

    public PiperNativeGraphPrefixRunner(long randomSeed) {
        this.randomSeed = randomSeed;
    }

    public PrefixResult execute(PiperModelImport imported, PiperSynthesisInput input, int nodeLimit)
            throws IOException {
        Objects.requireNonNull(imported, "imported");
        Objects.requireNonNull(input, "input");
        List<OnnxNode> nodes = imported.graph().nodes();
        if (nodeLimit < 1 || nodeLimit > nodes.size()) {
            throw new IllegalArgumentException("Piper graph prefix length is outside model bounds");
        }
        Map<String, PiperTensor> active = new HashMap<>();
        active.put("input", PiperTensor.integers(input.phonemeIds(),
                1, input.phonemeCount()));
        active.put("input_lengths", PiperTensor.integers(input.phonemeLengths(), 1));
        active.put("scales", PiperTensor.floats(input.scales(), 3));
        Map<String, OnnxTensor> initializers = imported.graph().initializers();
        PiperTensor last = null;
        java.util.Random random = new java.util.Random(randomSeed);

        for (int index = 0; index < nodeLimit; index++) {
            OnnxNode node = nodes.get(index);
            if ("Split".equals(node.opType())) {
                List<PiperTensor> pieces = split(node, active, initializers);
                if (pieces.size() != node.outputs().size()) {
                    throw new IllegalArgumentException("Piper Split output count mismatch at node " + index);
                }
                for (int part = 0; part < pieces.size(); part++) {
                    active.put(node.outputs().get(part), pieces.get(part));
                }
                last = pieces.get(pieces.size() - 1);
                continue;
            }
            try {
                last = evaluate(node, active, initializers, random);
            } catch (UnsupportedOperationException unsupported) {
                throw new UnsupportedOperationException("Piper native reference does not support "
                        + node.opType() + " at node " + index, unsupported);
            } catch (IllegalArgumentException problem) {
                throw new IllegalArgumentException("Piper graph node " + index
                        + " (" + node.opType() + ") failed: " + problem.getMessage(), problem);
            }
            if (node.outputs().size() != 1) {
                throw new UnsupportedOperationException("Piper prefix cannot execute multi-output "
                        + node.opType() + " at node " + index);
            }
            active.put(node.outputs().get(0), last);
        }
        return new PrefixResult(nodeLimit, nodes.get(nodeLimit - 1).opType(), last);
    }

    private static PiperTensor evaluate(OnnxNode node, Map<String, PiperTensor> active,
                                        Map<String, OnnxTensor> initializers,
                                        java.util.Random random) throws IOException {
        String operator = node.opType();
        return switch (operator) {
            case "GatherND" -> PiperIndexOperations.gatherNd(input(node, 0, active, initializers),
                    input(node, 1, active, initializers));
            case "ScatterND" -> PiperIndexOperations.scatterNd(input(node, 0, active, initializers),
                    input(node, 1, active, initializers),
                    input(node, 2, active, initializers));
            case "GatherElements" -> PiperIndexOperations.gatherElements(
                    input(node, 0, active, initializers),
                    input(node, 1, active, initializers),
                    Math.toIntExact(node.getInt("axis", 0)));
            case "NonZero" -> PiperIndexOperations.nonZero(input(node, 0, active, initializers));
            case "Gather" -> PiperTensorOperations.gather(
                    input(node, 0, active, initializers),
                    input(node, 1, active, initializers),
                    Math.toIntExact(node.getInt("axis", 0)));
            case "Mul", "Add", "Sub", "Div", "Pow", "Less",
                    "GreaterOrEqual", "LessOrEqual", "And" -> PiperTensorOperations.binary(operator,
                    input(node, 0, active, initializers), input(node, 1, active, initializers));
            case "Transpose" -> PiperTensorOperations.transpose(
                    input(node, 0, active, initializers), node.getInts("perm"));
            case "Shape" -> PiperTensorOperations.shape(input(node, 0, active, initializers));
            case "ConstantOfShape" -> PiperShapeOperations.constantOfShape(
                    input(node, 0, active, initializers), node.getTensor("value"));
            case "RandomNormalLike" -> PiperTensorOperations.randomNormalLike(
                    input(node, 0, active, initializers),
                    Math.toIntExact(node.getInt("dtype", PiperTensor.FLOAT32)), random);
            case "Expand" -> PiperShapeOperations.expand(input(node, 0, active, initializers),
                    input(node, 1, active, initializers));
            case "Squeeze" -> PiperShapeOperations.squeeze(input(node, 0, active, initializers),
                    node.inputs().size() > 1 && !node.inputs().get(1).isEmpty()
                            ? input(node, 1, active, initializers) : null);
            case "Not" -> PiperTensorOperations.not(input(node, 0, active, initializers));
            case "CumSum" -> PiperReductionOperations.cumSum(
                    input(node, 0, active, initializers),
                    input(node, 1, active, initializers),
                    node.getInt("exclusive", 0) != 0,
                    node.getInt("reverse", 0) != 0);
            case "ReduceSum", "ReduceMax" -> reduce(node, active, initializers);
            case "ReduceMean" -> PiperTensorOperations.reduceMean(
                    input(node, 0, active, initializers), node.getInts("axes"),
                    node.getInt("keepdims", 1) != 0);
            case "Relu", "Sqrt", "Erf", "Exp", "Tanh", "Sigmoid", "Neg" -> PiperTensorOperations.unary(
                    operator, input(node, 0, active, initializers));
            case "Softplus" -> PiperActivationOperations.softplus(
                    input(node, 0, active, initializers));
            case "Ceil" -> PiperActivationOperations.ceil(
                    input(node, 0, active, initializers));
            case "LeakyRelu" -> PiperActivationOperations.leakyRelu(
                    input(node, 0, active, initializers),
                    ((Number) node.attrs().getOrDefault("alpha", 0.01f)).floatValue());
            case "Clip" -> clip(node, active, initializers);
            case "Range" -> PiperTensorOperations.range(input(node, 0, active, initializers),
                    input(node, 1, active, initializers), input(node, 2, active, initializers));
            case "Unsqueeze" -> PiperTensorOperations.unsqueeze(
                    input(node, 0, active, initializers), input(node, 1, active, initializers));
            case "Cast" -> PiperTensorOperations.cast(input(node, 0, active, initializers),
                    Math.toIntExact(node.getInt("to", -1)));
            case "Concat" -> concatenate(node, active, initializers);
            case "Reshape" -> PiperTensorOperations.reshapeOnnx(
                    input(node, 0, active, initializers),
                    input(node, 1, active, initializers), node.getInt("allowzero", 0) != 0);
            case "MatMul" -> PiperTensorOperations.matmul(input(node, 0, active, initializers),
                    input(node, 1, active, initializers));
            case "Equal" -> PiperTensorOperations.equal(input(node, 0, active, initializers),
                    input(node, 1, active, initializers));
            case "Where" -> PiperTensorOperations.where(input(node, 0, active, initializers),
                    input(node, 1, active, initializers),
                    input(node, 2, active, initializers));
            case "Softmax" -> PiperTensorOperations.softmax(input(node, 0, active, initializers),
                    Math.toIntExact(node.getInt("axis", -1)));
            case "Slice" -> slice(node, active, initializers);
            case "Pad" -> zeroPad(node, active, initializers);
            case "Conv", "ConvTranspose" -> convolution(node, active, initializers);
            default -> throw new UnsupportedOperationException(operator);
        };
    }

    private static PiperTensor clip(OnnxNode node, Map<String, PiperTensor> active,
                                    Map<String, OnnxTensor> initializers) throws IOException {
        PiperTensor lower = node.inputs().size() > 1 && !node.inputs().get(1).isEmpty()
                ? input(node, 1, active, initializers) : null;
        PiperTensor upper = node.inputs().size() > 2 && !node.inputs().get(2).isEmpty()
                ? input(node, 2, active, initializers) : null;
        return PiperActivationOperations.clip(input(node, 0, active, initializers), lower, upper);
    }

    private static PiperTensor reduce(OnnxNode node, Map<String, PiperTensor> active,
                                      Map<String, OnnxTensor> initializers) throws IOException {
        long[] axes = node.inputs().size() > 1 && !node.inputs().get(1).isEmpty()
                ? input(node, 1, active, initializers).requireInteger()
                : node.getInts("axes").stream().mapToLong(Long::longValue).toArray();
        PiperTensor source = input(node, 0, active, initializers);
        boolean keepDimensions = node.getInt("keepdims", 1) != 0;
        boolean noop = node.getInt("noop_with_empty_axes", 0) != 0;
        return "ReduceMax".equals(node.opType())
                ? PiperReductionOperations.reduceMax(source, axes, keepDimensions, noop)
                : PiperReductionOperations.reduceSum(source, axes, keepDimensions, noop);
    }

    private static PiperTensor slice(OnnxNode node, Map<String, PiperTensor> active,
                                     Map<String, OnnxTensor> initializers) throws IOException {
        PiperTensor start = input(node, 1, active, initializers);
        PiperTensor end = input(node, 2, active, initializers);
        int count = start.length();
        long[] defaultAxes = new long[count];
        long[] defaultSteps = new long[count];
        for (int i = 0; i < count; i++) {
            defaultAxes[i] = i;
            defaultSteps[i] = 1L;
        }
        PiperTensor axes = node.inputs().size() > 3 && !node.inputs().get(3).isEmpty()
                ? input(node, 3, active, initializers)
                : PiperTensor.integers(defaultAxes, count);
        PiperTensor steps = node.inputs().size() > 4 && !node.inputs().get(4).isEmpty()
                ? input(node, 4, active, initializers)
                : PiperTensor.integers(defaultSteps, count);
        return PiperTensorOperations.slice(input(node, 0, active, initializers),
                start, end, axes, steps);
    }

    private static List<PiperTensor> split(OnnxNode node, Map<String, PiperTensor> active,
                                          Map<String, OnnxTensor> initializers) throws IOException {
        long[] sizes = null;
        if (node.inputs().size() > 1 && !node.inputs().get(1).isEmpty()) {
            sizes = input(node, 1, active, initializers).requireInteger();
        } else if (!node.getInts("split").isEmpty()) {
            sizes = node.getInts("split").stream().mapToLong(Long::longValue).toArray();
        }
        return PiperTensorOperations.split(input(node, 0, active, initializers),
                Math.toIntExact(node.getInt("axis", 0)), sizes, node.outputs().size());
    }

    private static PiperTensor zeroPad(OnnxNode node, Map<String, PiperTensor> active,
                                       Map<String, OnnxTensor> initializers) throws IOException {
        String mode = node.getString("mode", "constant");
        if (!"constant".equals(mode)) {
            throw new UnsupportedOperationException("Only constant-zero VITS padding is supported");
        }
        if (node.inputs().size() > 2 && !node.inputs().get(2).isEmpty()) {
            PiperTensor value = input(node, 2, active, initializers);
            if (value.length() != 1 || (value.dataType() == PiperTensor.FLOAT32
                    ? value.floatData()[0] != 0 : value.integerData()[0] != 0)) {
                throw new UnsupportedOperationException("Nonzero VITS padding is not yet supported");
            }
        }
        return PiperTensorOperations.pad(input(node, 0, active, initializers),
                input(node, 1, active, initializers));
    }

    private static PiperTensor concatenate(OnnxNode node, Map<String, PiperTensor> active,
                                           Map<String, OnnxTensor> initializers) throws IOException {
        List<PiperTensor> inputs = new ArrayList<>();
        for (int index = 0; index < node.inputs().size(); index++) {
            inputs.add(input(node, index, active, initializers));
        }
        return PiperTensorOperations.concat(inputs, Math.toIntExact(node.getInt("axis", 0)));
    }

    private static PiperTensor convolution(OnnxNode node, Map<String, PiperTensor> active,
                                           Map<String, OnnxTensor> initializers) throws IOException {
        PiperTensor bias = node.inputs().size() > 2 && !node.inputs().get(2).isEmpty()
                ? input(node, 2, active, initializers) : null;
        return PiperConvolutionExecutor.apply(node,
                input(node, 0, active, initializers),
                input(node, 1, active, initializers), bias);
    }

    private static PiperTensor input(OnnxNode node, int index, Map<String, PiperTensor> active,
                                     Map<String, OnnxTensor> initializers) throws IOException {
        String name = node.inputs().get(index);
        PiperTensor tensor = active.get(name);
        if (tensor != null) {
            return tensor;
        }
        OnnxTensor initialized = initializers.get(name);
        if (initialized != null) {
            tensor = PiperTensor.fromOnnx(initialized);
            active.put(name, tensor);
            return tensor;
        }
        throw new IllegalArgumentException("Unknown input tensor " + name);
    }

    public record PrefixResult(int executedNodes, String lastOperator, PiperTensor output) {
        public PrefixResult {
            Objects.requireNonNull(lastOperator, "lastOperator");
            Objects.requireNonNull(output, "output");
        }
    }
}
