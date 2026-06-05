package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.windows.WindowsBindings;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Creates and owns WARP/DirectML projections for a T5 pipeline.
 */
public final class T5WarpLinearProjectionFactory implements T5LinearProjectionFactory {
    private final WindowsBindings windowsBindings;
    private final List<T5LinearProjection> projections = new ArrayList<>();
    private boolean closed;

    public T5WarpLinearProjectionFactory(WindowsBindings windowsBindings) {
        this.windowsBindings = Objects.requireNonNull(windowsBindings, "windowsBindings");
    }

    @Override
    public T5LinearProjection create(T5TensorData weight) {
        ensureOpen();
        T5LinearProjection projection = T5WarpLinearProjection.from(windowsBindings, weight);
        projections.add(projection);
        return projection;
    }

    public int projectionCount() {
        return projections.size();
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("T5 WARP projection factory is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            RuntimeException failure = null;
            for (T5LinearProjection projection : projections) {
                try {
                    projection.close();
                } catch (Exception e) {
                    RuntimeException closeFailure = new RuntimeException(
                            "Failed to close T5 WARP projection: " + projection.name(), e);
                    if (failure == null) {
                        failure = closeFailure;
                    } else {
                        failure.addSuppressed(closeFailure);
                    }
                }
            }
            projections.clear();
            if (failure != null) {
                throw failure;
            }
        }
    }
}
