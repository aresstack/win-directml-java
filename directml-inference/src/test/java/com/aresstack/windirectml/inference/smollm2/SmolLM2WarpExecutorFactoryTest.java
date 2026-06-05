package com.aresstack.windirectml.inference.smollm2;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SmolLM2WarpExecutorFactoryTest {

    @AfterEach
    void clearProperty() {
        System.clearProperty(SmolLM2WarpExecutorFactory.EXECUTOR_CLASS_PROPERTY);
    }

    @Test
    void returnsExplicitUnsupportedExecutorWhenNoNativeExecutorIsConfigured() {
        SmolLM2WarpExecutor executor = SmolLM2WarpExecutorFactory.createDefaultExecutor();
        SmolLM2WarpExecutionStatus status = executor.inspect(null);

        assertFalse(status.executable());
        assertTrue(status.reason().contains(SmolLM2WarpExecutorFactory.EXECUTOR_CLASS_PROPERTY));
        assertFalse(status.warnings().isEmpty());
    }

    @Test
    void createsConfiguredExecutorFromSystemProperty() {
        System.setProperty(SmolLM2WarpExecutorFactory.EXECUTOR_CLASS_PROPERTY,
                SmolLM2ConfiguredWarpExecutor.class.getName());

        SmolLM2WarpExecutor executor = SmolLM2WarpExecutorFactory.createDefaultExecutor();
        SmolLM2WarpExecutionStatus status = executor.inspect(null);

        assertTrue(status.executable());
        assertEquals("configured test executor is available", status.reason());
    }

    @Test
    void reportsInvalidConfiguredExecutorClass() {
        System.setProperty(SmolLM2WarpExecutorFactory.EXECUTOR_CLASS_PROPERTY, String.class.getName());

        SmolLM2WarpExecutionStatus status = SmolLM2WarpExecutorFactory.createDefaultExecutor().inspect(null);

        assertFalse(status.executable());
        assertTrue(status.reason().contains("does not implement"));
    }
}
