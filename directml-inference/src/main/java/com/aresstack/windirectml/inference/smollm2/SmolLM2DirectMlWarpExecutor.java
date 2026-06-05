package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.runtime.DirectMlContextImpl;
import com.aresstack.windirectml.runtime.GpuBuffer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Built-in DirectML/WARP readiness probe for the future SmolLM2 WARP executor.
 *
 * <p>This executor intentionally does not report executable SmolLM2 inference yet. It only verifies that the
 * DirectML/WARP context can be initialized and that a small GPU buffer can be allocated. Real SmolLM2 kernels still
 * need a dedicated native executor implementation.</p>
 */
public final class SmolLM2DirectMlWarpExecutor implements SmolLM2WarpExecutor {

    private static final long PROBE_BUFFER_BYTES = 256L;
    private static final Pattern MEMORY_SEGMENT_ADDRESS_PATTERN = Pattern.compile("address:([0-9]+)");

    private final String backend;

    public SmolLM2DirectMlWarpExecutor() {
        this("warp");
    }

    public SmolLM2DirectMlWarpExecutor(String backend) {
        this.backend = backend == null || backend.isBlank() ? "warp" : backend.trim();
    }

    @Override
    public SmolLM2WarpExecutionStatus inspect(SmolLM2WarpRuntimePlan plan) {
        List<String> warnings = new ArrayList<>();
        if (plan != null) {
            warnings.addAll(plan.warnings());
        }

        try (DirectMlContextImpl context = new DirectMlContextImpl(backend)) {
            context.initialize();
            try (GpuBuffer ignored = context.allocateBuffer(PROBE_BUFFER_BYTES, GpuBuffer.BufferUsage.ACTIVATION)) {
                String reason = "SmolLM2 DirectML/WARP probe succeeded on "
                        + describeAdapter(context.adapterDescription())
                        + ", but SmolLM2 WARP kernels are not implemented yet.";
                return new SmolLM2WarpExecutionStatus(false, "warp", reason, warnings);
            }
        } catch (RuntimeException e) {
            String reason = "SmolLM2 DirectML/WARP probe failed for backend=" + backend + ": " + safeMessage(e);
            return new SmolLM2WarpExecutionStatus(false, "warp", reason, warnings);
        }
    }

    @Override
    public SmolLM2TokenRuntimeResult generate(SmolLM2Weights weights,
                                              SmolLM2WarpRuntimePlan plan,
                                              SmolLM2TokenRuntimeRequest request) {
        Objects.requireNonNull(weights, "weights");
        Objects.requireNonNull(plan, "plan");
        Objects.requireNonNull(request, "request");
        throw new SmolLM2RuntimeUnsupportedException(
                "SmolLM2 DirectML/WARP probe is available, but SmolLM2 WARP kernels are not implemented yet.");
    }

    private static String describeAdapter(String adapterDescription) {
        if (adapterDescription == null || adapterDescription.isBlank()) {
            return "DirectML/WARP adapter";
        }

        Matcher matcher = MEMORY_SEGMENT_ADDRESS_PATTERN.matcher(adapterDescription);
        if (!matcher.find()) {
            return adapterDescription;
        }

        try {
            long address = Long.parseUnsignedLong(matcher.group(1));
            String suffix = adapterDescription.contains("(DirectML)")
                    ? " (DirectML)"
                    : adapterDescription.contains("(D3D12 only)") ? " (D3D12 only)" : "";
            return "DXGI adapter pointer=0x" + Long.toUnsignedString(address, 16) + suffix;
        } catch (NumberFormatException ignored) {
            return "DXGI adapter pointer=<unavailable>";
        }
    }

    private static String safeMessage(RuntimeException exception) {
        if (exception.getMessage() == null || exception.getMessage().isBlank()) {
            return exception.getClass().getSimpleName();
        }
        return exception.getMessage();
    }
}
