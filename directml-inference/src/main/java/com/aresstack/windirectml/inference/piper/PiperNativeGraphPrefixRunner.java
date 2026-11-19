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
                last = evaluate(node, active, initializers);
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
                                        Map<String, OnnxTensor> initializers) throws IOException {
        String operator = node.opType();
        return switch (operator) {
            case "Gather" -> PiperTensorOperations.gather(
                    input(node, 0, active, initializers),
                    input(node, 1, active, initializers),
                    Math.toIntExact(node.getInt("axis", 0)));
            case "Mul", "Add", "Sub", "Div", "Pow", "Less" -> PiperTensorOperations.binary(operator,
                    input(node, 0, active, initializers), input(node, 1, active, initializers));
            case "Transpose" -> PiperTensorOperations.transpose(
                    input(node, 0, active, initializers), node.getInts("perm"));
            case "Shape" -> PiperTensorOperations.shape(input(node, 0, active, initializers));
            case "ReduceMean" -> PiperTensorOperations.reduceMean(
                    input(node, 0, active, initializers), node.getInts("axes"),
                    node.getInt("keepdims", 1) != 0);
            case "Relu", "Sqrt", "Erf", "Exp", "Tanh", "Sigmoid", "Neg" -> PiperTensorOperations.unary(
                    operator, input(node, 0, active, initializers));
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
            case "Slice" -> PiperTensorOperations.slice(
                    input(node, 0, active, initializers),
                    input(node, 1, active, initializers),
                    input(node, 2, active, initializers),
                    input(node, 3, active, initializers),
                    input(node, 4, active, initializers));
            case "Pad" -> zeroPad(node, active, initializers);
            case "Conv" -> convolution(node, active, initializers);
            default -> throw new UnsupportedOperationException(operator);
        };
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
        PiperTensor source = input(node, 0, active, initializers);
        PiperTensor filters = input(node, 1, active, initializers);
        PiperTensor bias = node.inputs().size() > 2 && !node.inputs().get(2).isEmpty()
                ? input(node, 2, active, initializers) : null;
        int[] xShape = source.shape();
        int[] wShape = filters.shape();
        if (xShape.length != 3 || wShape.length != 3) {
            throw new IllegalArgumentException("Only 1-D NCL Piper convolution is implemented");
        }
        int batches = xShape[0];
        int inputs = xShape[1];
        int width = xShape[2];
        int outputs = wShape[0];
        int channelsPerGroup = wShape[1];
        int kernel = wShape[2];
        int groupCount = Math.toIntExact(node.getInt("group", 1));
        int stride = parameter(node, "strides", 1);
        int dilation = parameter(node, "dilations", 1);
        List<Long> pads = node.getInts("pads");
        int left = pads.isEmpty() ? 0 : Math.toIntExact(pads.get(0));
        int right = pads.isEmpty() ? 0 : Math.toIntExact(pads.get(1));
        float[] sourceValues = source.requireFloat();
        float[] weightValues = filters.requireFloat();
        float[] biases = bias == null ? null : bias.requireFloat();

        float[][][] weights = new float[outputs][channelsPerGroup][kernel];
        for (int channel = 0; channel < outputs; channel++) {
            for (int sourceChannel = 0; sourceChannel < channelsPerGroup; sourceChannel++) {
                System.arraycopy(weightValues,
                        (channel * channelsPerGroup + sourceChannel) * kernel,
                        weights[channel][sourceChannel], 0, kernel);
            }
        }
        float[] flattened = null;
        int outputWidth = 0;
        for (int batch = 0; batch < batches; batch++) {
            float[][] x = new float[inputs][width];
            for (int channel = 0; channel < inputs; channel++) {
                System.arraycopy(sourceValues, (batch * inputs + channel) * width,
                        x[channel], 0, width);
            }
            float[][] y = PiperConv1dReference.apply(x, weights, biases,
                    stride, dilation, left, right, groupCount);
            if (flattened == null) {
                outputWidth = y[0].length;
                flattened = new float[Math.multiplyExact(Math.multiplyExact(batches, outputs), outputWidth)];
            }
            for (int channel = 0; channel < outputs; channel++) {
                System.arraycopy(y[channel], 0, flattened,
                        (batch * outputs + channel) * outputWidth, outputWidth);
            }
        }
        return PiperTensor.floats(flattened, batches, outputs, outputWidth);
    }

    private static int parameter(OnnxNode node, String name, int defaultValue) {
        List<Long> values = node.getInts(name);
        return values.isEmpty() ? defaultValue : Math.toIntExact(values.get(0));
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
