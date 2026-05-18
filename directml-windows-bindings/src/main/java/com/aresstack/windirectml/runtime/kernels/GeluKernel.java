package com.aresstack.windirectml.runtime.kernels;

import com.aresstack.windirectml.runtime.DirectMlContextImpl;
import com.aresstack.windirectml.runtime.DirectMlTensor;
import com.aresstack.windirectml.runtime.DirectMlRuntimeException;
import com.aresstack.windirectml.windows.DirectMlBindings;

/**
 * GELU-Aktivierung (elementweise). Verwendet von BERT, MiniLM, JinaBERT.
 * <p>
 * Zwei Implementierungen:
 * <ul>
 *   <li>{@link DirectMlGeluKernel} – nativer fused
 *       {@link DirectMlBindings#DML_OPERATOR_ACTIVATION_GELU} (op 157,
 *       {@code DML_FEATURE_LEVEL_5_1}+).</li>
 *   <li>{@link DirectMlCompositeGeluKernel} – Fallback aus {@code ERF +
 *       IDENTITY + MULTIPLY} (FL 1.0/2.0), läuft auf jeder shippenden
 *       {@code DirectML.dll} – inkl. Windows-11-In-Box mit FL 5.0.</li>
 * </ul>
 * Die Pipeline ruft {@link #create(DirectMlContextImpl, int)} auf und
 * bekommt automatisch die für das Feature-Level richtige Variante.
 */
public interface GeluKernel extends AutoCloseable {

    void dispatch(DirectMlTensor x, DirectMlTensor y) throws DirectMlRuntimeException;

    @Override
    void close();

    /**
     * Erzeugt die für das Feature-Level des laufenden Context passende
     * GELU-Variante: nativ bei FL ≥ 5.1, sonst Composite.
     */
    static GeluKernel create(DirectMlContextImpl ctx, int elementCount)
            throws DirectMlRuntimeException {
        if (ctx == null || !ctx.isReady()) {
            throw new DirectMlRuntimeException("Context not ready");
        }
        int fl = ctx.bindings().getDmlFeatureLevel();
        if (DirectMlBindings.supportsFusedGelu(fl)) {
            return new DirectMlGeluKernel(ctx, elementCount);
        }
        return new DirectMlCompositeGeluKernel(ctx, elementCount);
    }
}

