package com.aresstack.windirectml.inference.warp;

import com.aresstack.windirectml.windows.MatMulNBitsKernel;
import com.aresstack.windirectml.windows.WarpExecutionContext;
import com.aresstack.windirectml.windows.WarpGpuBuffer;
import com.aresstack.windirectml.windows.WindowsBindings;
import com.aresstack.windirectml.windows.WindowsNativeException;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Objects;

/**
 * Model-family-neutral WARP/DirectML dense (matvec/GEMV) projection.
 *
 * <p>The single shared adapter over the DirectML {@link MatMulNBitsKernel} for {@code y = W * x} with a rank-2 weight
 * matrix shaped {@code [output, input]}. It replaces per-family copies (T5's {@code T5WarpLinearProjection}, and —
 * later, as a separate migration — the decoder-only WARP dense projection) so all families share one upload + one
 * matvec/batch path.</p>
 *
 * <p>The API is intentionally the union of what the families need: an allocating and an into-buffer matvec, an
 * allocating and an into-buffer batched sequence projection (with a per-row fallback when batching is unavailable),
 * and access to the underlying {@link #kernel()} for GPU-resident subgraph chaining. The weights are uploaded once at
 * construction and reused.</p>
 */
public final class WarpDenseProjection implements AutoCloseable {

    private final String name;
    private final int inputSize;
    private final int outputSize;
    private final MatMulNBitsKernel kernel;
    private boolean batchDisabled;
    private boolean closed;

    private WarpDenseProjection(String name, int inputSize, int outputSize, MatMulNBitsKernel kernel) {
        this.name = Objects.requireNonNull(name, "name");
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.kernel = Objects.requireNonNull(kernel, "kernel");
    }

    /**
     * Build a projection from dequantized row-major {@code [output, input]} weights. The shape is validated before any
     * native kernel is created.
     */
    public static WarpDenseProjection fromDequantizedWeights(WindowsBindings windowsBindings,
                                                             String name,
                                                             int outputSize,
                                                             int inputSize,
                                                             float[] weights) {
        // Validate the (pure) shape first so shape errors fail fast without needing native bindings.
        validateShape(name, outputSize, inputSize, weights);
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        MatMulNBitsKernel kernel = MatMulNBitsKernel.fromDequantizedWeights(
                windowsBindings, outputSize, inputSize, weights);
        return new WarpDenseProjection(name, inputSize, outputSize, kernel);
    }

    /**
     * Heap-light variant of {@link #fromDequantizedWeights(WindowsBindings, String, int, int, float[])}: build a
     * projection from dequantized row-major {@code [output, input]} FP32 weights supplied as a raw little-endian
     * {@link ByteBuffer}, without first materialising a host {@code float[]}.
     *
     * <p>Slice H2a upload seam — accepts read-only direct {@code MappedByteBuffer} slices (e.g. a {@code .wdmlpack}
     * payload) and heap buffers; the buffer's {@code [position, limit)} region is uploaded verbatim and its
     * position/limit are not modified. Numerically identical to the {@code float[]} overload for the same
     * little-endian FP32 bytes. The existing {@code float[]} path is unchanged.</p>
     */
    public static WarpDenseProjection fromDequantizedWeights(WindowsBindings windowsBindings,
                                                             String name,
                                                             int outputSize,
                                                             int inputSize,
                                                             ByteBuffer fp32WeightsLe) {
        // Validate the (pure) shape + endianness first so errors fail fast without needing native bindings.
        validateShape(name, outputSize, inputSize, fp32WeightsLe);
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        MatMulNBitsKernel kernel = MatMulNBitsKernel.fromDequantizedWeights(
                windowsBindings, outputSize, inputSize, fp32WeightsLe);
        return new WarpDenseProjection(name, inputSize, outputSize, kernel);
    }

    /**
     * Build a projection from a {@link WarpWeightSource} — the single shared seam that picks the heap-light FP32
     * ByteBuffer upload when available and otherwise the {@code float[]} fallback. Families (T5, SmolLM2, …) build a
     * {@code WarpWeightSource} instead of duplicating the "ByteBuffer if present, else float[]" decision.
     */
    public static WarpDenseProjection fromWeightSource(WindowsBindings windowsBindings, WarpWeightSource source) {
        Objects.requireNonNull(source, "source");
        ByteBuffer fp32 = source.fp32LittleEndian();
        if (fp32 != null) {
            return fromDequantizedWeights(windowsBindings, source.name(),
                    source.outputRows(), source.inputColumns(), fp32);
        }
        return fromDequantizedWeights(windowsBindings, source.name(),
                source.outputRows(), source.inputColumns(), source.dequantizedRowMajor());
    }

    /**
     * Build a fused projection ({@code [outputSize, inputSize]} = several vertically-stacked FP32 parts) from raw
     * little-endian {@link ByteBuffer} part slices, without a host {@code float[]} concatenation (slice item 3). Each
     * part is uploaded directly into its row-region of the device weight buffer. Numerically identical to building the
     * same matrix from one concatenated FP32 array.
     */
    public static WarpDenseProjection fromFusedFp32(WindowsBindings windowsBindings, String name,
                                                    int outputSize, int inputSize, List<ByteBuffer> partsLe) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        Objects.requireNonNull(partsLe, "partsLe");
        if (outputSize < 1 || inputSize < 1) {
            throw new IllegalArgumentException("fused projection " + name + " dims must be positive: "
                    + outputSize + "x" + inputSize);
        }
        MatMulNBitsKernel kernel = MatMulNBitsKernel.fromFusedFp32ByteBuffers(
                windowsBindings, outputSize, inputSize, partsLe);
        return new WarpDenseProjection(name, inputSize, outputSize, kernel);
    }

    public String name() {
        return name;
    }

    public int inputSize() {
        return inputSize;
    }

    public int outputSize() {
        return outputSize;
    }

    /**
     * The underlying GPU matmul kernel, for GPU-resident subgraphs that chain this projection's dispatch into a shared
     * pipeline without an intermediate CPU readback.
     */
    public MatMulNBitsKernel kernel() {
        return kernel;
    }

    /**
     * GPU-resident projection (GEMMA-WARP-13b-3a): {@code y = W * x} reading {@code input} from a resident
     * buffer and returning a new resident {@link WarpGpuBuffer} — no CPU upload/readback. Numerically
     * identical to {@link #project(float[])}.
     */
    public WarpGpuBuffer forwardResident(WarpExecutionContext ctx, WarpGpuBuffer input)
            throws WindowsNativeException {
        ensureOpen();
        Objects.requireNonNull(ctx, "ctx");
        if (input.elementCount() != inputSize) {
            throw new IllegalArgumentException("WARP dense projection resident input length mismatch for "
                    + name + ": input=" + input.elementCount() + ", expected=" + inputSize);
        }
        WarpGpuBuffer output = ctx.allocate(outputSize);
        // Routes through the context: direct-UAV into the open recording list (GEMMA-WARP-13d) when one is
        // active, else the kernel's standalone resident matvec. Same math either way.
        ctx.matvec(kernel, input, output);
        return output;
    }

    /** Project one vector, allocating the output. */
    public float[] project(float[] input) {
        float[] output = new float[outputSize];
        projectInto(input, output);
        return output;
    }

    /** Project one vector into {@code output[0..outputSize)}. */
    public void projectInto(float[] input, float[] output) {
        ensureOpen();
        validateInput(input);
        Objects.requireNonNull(output, "output");
        if (output.length < outputSize) {
            throw new IllegalArgumentException("WARP dense projection output buffer too small for " + name
                    + ": output=" + output.length + ", expected at least=" + outputSize);
        }
        kernel.matvec(input, output);
    }

    /** Project a row-major sequence of {@code sequenceLength} vectors, allocating the output. */
    public float[] projectSequence(float[] input, int sequenceLength) {
        if (sequenceLength < 1) {
            throw new IllegalArgumentException("sequenceLength must be positive for " + name + ": " + sequenceLength);
        }
        float[] output = new float[Math.multiplyExact(sequenceLength, outputSize)];
        projectSequenceInto(input, sequenceLength, output);
        return output;
    }

    /** Project a row-major sequence into {@code output}; batched when supported, otherwise per row. */
    public void projectSequenceInto(float[] input, int sequenceLength, float[] output) {
        ensureOpen();
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        if (sequenceLength < 1) {
            throw new IllegalArgumentException("sequenceLength must be positive for " + name + ": " + sequenceLength);
        }
        if (input.length != sequenceLength * inputSize) {
            throw new IllegalArgumentException("WARP dense projection sequence input length mismatch for " + name
                    + ": values=" + input.length + ", expected=" + (sequenceLength * inputSize));
        }
        if (output.length < sequenceLength * outputSize) {
            throw new IllegalArgumentException("WARP dense projection sequence output too small for " + name
                    + ": output=" + output.length + ", expected at least=" + (sequenceLength * outputSize));
        }
        if (sequenceLength == 1 || batchDisabled || !kernel.supportsBatch()) {
            projectRows(input, output, sequenceLength);
            return;
        }
        try {
            kernel.matmulBatch(input, output, sequenceLength);
        } catch (RuntimeException ex) {
            batchDisabled = true;
            projectRows(input, output, sequenceLength);
        }
    }

    private void projectRows(float[] input, float[] output, int sequenceLength) {
        float[] row = new float[inputSize];
        float[] rowOutput = new float[outputSize];
        for (int token = 0; token < sequenceLength; token++) {
            System.arraycopy(input, token * inputSize, row, 0, inputSize);
            kernel.matvec(row, rowOutput);
            System.arraycopy(rowOutput, 0, output, token * outputSize, outputSize);
        }
    }

    public boolean isClosed() {
        return closed;
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            kernel.close();
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("WARP dense projection is closed: " + name);
        }
    }

    private void validateInput(float[] input) {
        Objects.requireNonNull(input, "input");
        if (input.length != inputSize) {
            throw new IllegalArgumentException("WARP dense projection input length mismatch for " + name
                    + ": input=" + input.length + ", expected=" + inputSize);
        }
    }

    private static void validateShape(String name, int outputSize, int inputSize, float[] weights) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(weights, "weights");
        if (outputSize < 1) {
            throw new IllegalArgumentException("outputSize must be positive for " + name + ": " + outputSize);
        }
        if (inputSize < 1) {
            throw new IllegalArgumentException("inputSize must be positive for " + name + ": " + inputSize);
        }
        long expected = (long) outputSize * inputSize;
        if (weights.length != expected) {
            throw new IllegalArgumentException("WARP dense projection weight length mismatch for " + name
                    + ": weights=" + weights.length + ", expected=" + expected);
        }
    }

    private static void validateShape(String name, int outputSize, int inputSize, ByteBuffer fp32WeightsLe) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(fp32WeightsLe, "fp32WeightsLe");
        if (outputSize < 1) {
            throw new IllegalArgumentException("outputSize must be positive for " + name + ": " + outputSize);
        }
        if (inputSize < 1) {
            throw new IllegalArgumentException("inputSize must be positive for " + name + ": " + inputSize);
        }
        if (fp32WeightsLe.order() != ByteOrder.LITTLE_ENDIAN) {
            throw new IllegalArgumentException("WARP dense projection ByteBuffer must be LITTLE_ENDIAN for "
                    + name + ", got " + fp32WeightsLe.order());
        }
        long expectedBytes = (long) outputSize * inputSize * Float.BYTES;
        if (fp32WeightsLe.remaining() != expectedBytes) {
            throw new IllegalArgumentException("WARP dense projection weight size mismatch for " + name
                    + ": remaining=" + fp32WeightsLe.remaining() + " bytes, expected=" + expectedBytes);
        }
    }
}
