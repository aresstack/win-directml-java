package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.windows.WindowsBindings;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Native DirectML/WARP executor for SmolLM2.
 *
 * <p>Runs the full SmolLM2 decoder stack with every dense projection on the shared
 * {@link com.aresstack.windirectml.inference.decoderonly.DecoderOnlyWarpDenseProjection} GPU kernel. Norms, RoPE,
 * grouped-query attention and the KV cache reuse the shared {@code decoderonly} math, so the WARP path is numerically
 * aligned with {@link SmolLM2ReferenceForwardPass} apart from GPU floating-point rounding.</p>
 *
 * <p>The executor builds a {@link SmolLM2NativeWarpSession} that uploads the projection weights once and reuses them
 * across decode steps. When the WARP device is unavailable it reports a non-executable status so {@code AUTO} falls
 * back to the CPU reference runtime.</p>
 */
public final class SmolLM2NativeWarpExecutor implements SmolLM2WarpExecutor {

    private final String backend;

    public SmolLM2NativeWarpExecutor() {
        this("warp");
    }

    public SmolLM2NativeWarpExecutor(String backend) {
        this.backend = backend == null || backend.isBlank() ? "warp" : backend.trim();
    }

    @Override
    public SmolLM2WarpSession openSession(SmolLM2Weights weights, SmolLM2WarpRuntimePlan plan) {
        Objects.requireNonNull(weights, "weights");
        Objects.requireNonNull(plan, "plan");
        return new SmolLM2NativeWarpSession(weights, plan, backend);
    }

    @Override
    public SmolLM2WarpExecutionStatus inspect(SmolLM2WarpRuntimePlan plan) {
        List<String> warnings = new ArrayList<>();
        if (plan != null) {
            warnings.addAll(plan.warnings());
        }
        if (!WindowsBindings.isSupported()) {
            String reason = "SmolLM2 native WARP executor unavailable: Windows D3D12/DirectML is not supported on this host.";
            warnings.add(reason);
            return new SmolLM2WarpExecutionStatus(false, backend, reason, warnings);
        }
        return new SmolLM2WarpExecutionStatus(true, backend,
                "SmolLM2 native WARP executor available (backend=" + backend + ").", warnings);
    }

    @Override
    public SmolLM2TokenRuntimeResult generate(SmolLM2Weights weights,
                                              SmolLM2WarpRuntimePlan plan,
                                              SmolLM2TokenRuntimeRequest request) {
        Objects.requireNonNull(weights, "weights");
        Objects.requireNonNull(plan, "plan");
        Objects.requireNonNull(request, "request");
        try (SmolLM2NativeWarpSession session = new SmolLM2NativeWarpSession(weights, plan, backend)) {
            if (!session.status().executable()) {
                throw new SmolLM2RuntimeUnsupportedException(session.status().reason());
            }
            return session.generateTokenIds(request);
        }
    }
}
