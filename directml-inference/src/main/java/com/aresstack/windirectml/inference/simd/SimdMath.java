package com.aresstack.windirectml.inference.simd;

/**
 * Optional-SIMD provider for {@link FloatMathOps}.
 *
 * <p>Resolves the best available implementation once, at class init, <b>without</b> referencing
 * {@code jdk.incubator.vector} directly: it loads the isolated {@code simd.vector.VectorFloatMathOps} reflectively. If
 * the incubator module is absent (no {@code --add-modules=jdk.incubator.vector}) or the Vector API is otherwise
 * unavailable, loading that class throws (e.g. {@code NoClassDefFoundError}); this class catches the {@link Throwable}
 * and falls back to {@link ScalarFloatMathOps}. Because {@code SimdMath} itself has no Vector-API references, it — and
 * its callers — load and run fine without the module.</p>
 */
public final class SimdMath {

    private static final String VECTOR_IMPL = "com.aresstack.windirectml.inference.simd.vector.VectorFloatMathOps";

    private static final FloatMathOps PROVIDER = resolve();

    private SimdMath() {
    }

    /** The resolved float-math implementation (SIMD when available, otherwise scalar). */
    public static FloatMathOps provider() {
        return PROVIDER;
    }

    private static FloatMathOps resolve() {
        try {
            Class<?> impl = Class.forName(VECTOR_IMPL);
            FloatMathOps ops = (FloatMathOps) impl.getDeclaredConstructor().newInstance();
            if (ops.enabled()) {
                return ops;
            }
        } catch (Throwable ignored) {
            // Vector module absent / Vector API unavailable -> scalar fallback.
        }
        return new ScalarFloatMathOps();
    }
}
