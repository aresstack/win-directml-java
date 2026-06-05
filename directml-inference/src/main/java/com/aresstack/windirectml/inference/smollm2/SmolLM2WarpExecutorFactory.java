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

    private SmolLM2WarpExecutorFactory() {
    }

    /**
     * Create the configured WARP executor or return an explicit unsupported executor.
     */
    public static SmolLM2WarpExecutor createDefaultExecutor() {
        String executorClassName = System.getProperty(EXECUTOR_CLASS_PROPERTY, "").trim();
        if (executorClassName.isEmpty()) {
            return new SmolLM2UnsupportedWarpExecutor("No SmolLM2 WARP executor is configured. Set -D"
                    + EXECUTOR_CLASS_PROPERTY
                    + "=<executor-class> after adding a native SmolLM2 WARP executor to the classpath.");
        }
        return createConfiguredExecutor(executorClassName);
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
