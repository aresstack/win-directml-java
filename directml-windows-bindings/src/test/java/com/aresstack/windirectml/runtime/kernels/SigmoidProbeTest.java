package com.aresstack.windirectml.runtime.kernels;

import com.aresstack.windirectml.runtime.DirectMlContextImpl;
import com.aresstack.windirectml.runtime.DirectMlRuntimeException;
import com.aresstack.windirectml.windows.DirectMlBindings;
import com.aresstack.windirectml.windows.DxgiBindings;
import com.aresstack.windirectml.windows.WindowsBindings;
import com.aresstack.windirectml.windows.WindowsNativeException;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Sanity-Check: bestätigt, dass die FFM-Infrastruktur (Tensor-Descs,
 * Op-Desc-Builder, CreateOperator-Vtable-Aufruf) korrekt funktioniert,
 * indem ein <em>FL 1.0</em>-Operator (SIGMOID, ID 47) gegen denselben
 * Code-Pfad gebaut wird, der für GELU fehlschlägt. Damit lässt sich
 * sauber unterscheiden:
 * <ul>
 *   <li>Schlägt SIGMOID auch fehl → Bug in unserem Code</li>
 *   <li>SIGMOID grün, GELU rot → DirectML-Runtime-Version zu alt</li>
 * </ul>
 */
class SigmoidProbeTest {

    @Test
    void sigmoidOperatorCanBeCreated() throws DirectMlRuntimeException {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12");

        DirectMlContextImpl ctx = new DirectMlContextImpl("directml");
        try {
            try {
                ctx.initialize();
            } catch (DirectMlRuntimeException e) {
                assumeTrue(false, "D3D12/DML stack not available: " + e.getMessage());
                return;
            }
            assumeTrue(ctx.bindings().hasDirectMl(), "No DirectML device");

            try (Arena arena = Arena.ofConfined()) {
                // Same builder pattern as DirectMlGeluKernel.
                MemorySegment buf = DirectMlBindings.allocBufferTensorDesc(arena,
                        DirectMlBindings.DML_TENSOR_DATA_TYPE_FLOAT32,
                        new int[]{1, 1, 1, 8}, null, 8L * Float.BYTES);
                MemorySegment xDesc = DirectMlBindings.allocTensorDesc(arena, buf);
                MemorySegment yDesc = DirectMlBindings.allocTensorDesc(arena, buf);

                // DML_ACTIVATION_SIGMOID_OPERATOR_DESC = { Input*, Output* }
                MemorySegment desc = arena.allocate(16, 8);
                desc.set(ValueLayout.ADDRESS, 0, xDesc);
                desc.set(ValueLayout.ADDRESS, 8, yDesc);

                final int DML_OPERATOR_ACTIVATION_SIGMOID = 47;
                MemorySegment opDesc = DirectMlBindings.allocOperatorDesc(arena,
                        DML_OPERATOR_ACTIVATION_SIGMOID, desc);

                MemorySegment dml = ctx.bindings().getDmlDevice();
                MemorySegment op;
                try {
                    op = DirectMlBindings.createOperator(dml, opDesc, arena);
                } catch (WindowsNativeException e) {
                    throw new AssertionError("SIGMOID (FL 1.0 op) failed – infrastructure bug", e);
                }
                assertNotNull(op, "operator handle must not be null");
                DxgiBindings.release(op);
            }
        } finally {
            ctx.close();
        }
    }
}

