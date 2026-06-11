package com.aresstack.windirectml.inference.smollm2;

import java.lang.reflect.Constructor;

/**
 * Creates the SmolLM2 WARP executor used by default runtime entry points.
 *
 * <p>The reference runtime remains the safe fallback. A native executor can be wired without changing the public
 * SmolLM2 runtime facade by setting {@value #EXECUTOR_CLASS_PROPERTY} to a class that implements
 * {@link SmolLM2WarpExecutor} and has a public no-argument constructor.</p>
 */
public final class SmolLM2WarpExecutorFactory {

    public static final String EXECUTOR_CLASS_PROPERTY = "windirectml.smollm2.warp.executorClass";
    public static final String EXECUTOR_MODE_PROPERTY = "windirectml.smollm2.warp.executorMode";

    private static final String EXECUTOR_MODE_NATIVE = "native";
    private static final String EXECUTOR_MODE_PROBE = "probe";
    private static final String EXECUTOR_MODE_NONE = "none";

    private SmolLM2WarpExecutorFactory() {
    }

    /**
     * Create the configured WARP executor.
     *
     * <p>By default this returns the {@link SmolLM2NativeWarpExecutor}, which runs the SmolLM2 decoder stack on the
     * shared decoder-only WARP projection kernels. The legacy DirectML readiness {@code probe} and {@code none}
     * modes remain available for diagnostics, and a fully custom executor can be supplied via
     * {@value #EXECUTOR_CLASS_PROPERTY}.</p>
     */
    public static SmolLM2WarpExecutor createDefaultExecutor() {
        return createDefaultExecutor("warp");
    }

    /**
     * Create the configured WARP executor bound to a specific D3D12 adapter backend.
     *
     * <p>{@code adapterBackend} selects the device the native executor initialises: {@code "warp"} uses the D3D12
     * WARP software rasterizer (CPU), {@code "directml"} uses the first hardware DirectML adapter. The math, kernels
     * and numerics are identical on both adapters — only the device differs. The dispatch/readback reductions of the
     * GPU-resident pipeline only pay off on a real hardware adapter, where each submission is an actual CPU↔GPU
     * round-trip; on WARP there is no round-trip to amortise.</p>
     */
    public static SmolLM2WarpExecutor createDefaultExecutor(String adapterBackend) {
        String executorClassName = System.getProperty(EXECUTOR_CLASS_PROPERTY, "").trim();
        if (!executorClassName.isEmpty()) {
            return createConfiguredExecutor(executorClassName);
        }

        String executorMode = System.getProperty(EXECUTOR_MODE_PROPERTY, EXECUTOR_MODE_NATIVE).trim();
        if (executorMode.isEmpty() || EXECUTOR_MODE_NATIVE.equalsIgnoreCase(executorMode)) {
            return new SmolLM2NativeWarpExecutor(adapterBackend);
        }
        if (EXECUTOR_MODE_PROBE.equalsIgnoreCase(executorMode)) {
            return new SmolLM2DirectMlWarpExecutor();
        }
        if (EXECUTOR_MODE_NONE.equalsIgnoreCase(executorMode)) {
            return new SmolLM2UnsupportedWarpExecutor("No SmolLM2 WARP executor is configured. Set -D"
                    + EXECUTOR_MODE_PROPERTY + "=native to run the native WARP executor, set -D"
                    + EXECUTOR_CLASS_PROPERTY + "=<executor-class> for a custom executor, "
                    + "or -D" + EXECUTOR_MODE_PROPERTY + "=probe for the DirectML readiness probe.");
        }
        return new SmolLM2UnsupportedWarpExecutor("Unsupported SmolLM2 WARP executor mode: " + executorMode
                + ". Supported values: native, probe, none. Use -D" + EXECUTOR_CLASS_PROPERTY
                + "=<executor-class> for a custom native executor.");
    }

    private static SmolLM2WarpExecutor createConfiguredExecutor(String executorClassName) {
        try {
            Class<?> executorClass = Class.forName(executorClassName);
            if (!SmolLM2WarpExecutor.class.isAssignableFrom(executorClass)) {
                return new SmolLM2UnsupportedWarpExecutor("Configured SmolLM2 WARP executor does not implement "
                        + SmolLM2WarpExecutor.class.getName() + ": " + executorClassName);
            }
            Constructor<?> constructor = executorClass.getDeclaredConstructor();
            if (!constructor.canAccess(null)) {
                constructor.setAccessible(true);
            }
            return (SmolLM2WarpExecutor) constructor.newInstance();
        } catch (ReflectiveOperationException | LinkageError ex) {
            return new SmolLM2UnsupportedWarpExecutor("Configured SmolLM2 WARP executor could not be created: "
                    + executorClassName + " (" + ex.getClass().getSimpleName() + ": "
                    + safeMessage(ex.getMessage()) + ")");
        }
    }

    private static String safeMessage(String message) {
        return message == null || message.isBlank() ? "no details" : message;
    }
}
