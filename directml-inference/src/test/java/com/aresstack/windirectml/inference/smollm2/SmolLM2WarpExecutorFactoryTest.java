package com.aresstack.windirectml.inference.smollm2;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertFalse;

final class SmolLM2WarpExecutorFactoryTest {

    @AfterEach
    void clearProperties() {
        System.clearProperty(SmolLM2WarpExecutorFactory.EXECUTOR_CLASS_PROPERTY);
        System.clearProperty(SmolLM2WarpExecutorFactory.EXECUTOR_MODE_PROPERTY);
    }

    @Test
    void createsBuiltInProbeExecutorByDefault() {
        assertInstanceOf(SmolLM2DirectMlWarpExecutor.class, SmolLM2WarpExecutorFactory.createDefaultExecutor());
    }

    @Test
    void canDisableProbeExecutorForExplicitUnsupportedMode() {
        System.setProperty(SmolLM2WarpExecutorFactory.EXECUTOR_MODE_PROPERTY, "none");

        assertInstanceOf(SmolLM2UnsupportedWarpExecutor.class, SmolLM2WarpExecutorFactory.createDefaultExecutor());
    }

    @Test
    void reportsUnsupportedForInvalidCustomExecutorClass() {
        System.setProperty(SmolLM2WarpExecutorFactory.EXECUTOR_CLASS_PROPERTY, "java.lang.String");

        SmolLM2WarpExecutor executor = SmolLM2WarpExecutorFactory.createDefaultExecutor();

        assertInstanceOf(SmolLM2UnsupportedWarpExecutor.class, executor);
        assertFalse(executor.inspect(null).executable());
    }
}
