package com.aresstack.windirectml.catalog;

import java.util.Locale;

/**
 * A backend a model family can run on. Encoders/rerankers ship a byte-faithful {@link #CPU} reference and a
 * {@link #DIRECTML} path; native generation families run on the {@link #WARP} software adapter and/or the
 * {@link #AUTO} hardware adapter. A family that only offers CPU through a non-native (e.g. external Python)
 * path deliberately does NOT list {@link #CPU} here, so a strict host never falls back to it.
 *
 * <p>Java-8 compatible.</p>
 */
public enum CatalogBackend {

    /** Native Java CPU reference runtime. */
    CPU,
    /** DirectML on a DirectX-12 GPU. */
    DIRECTML,
    /** The D3D12 WARP software rasterizer adapter (native, no Python). */
    WARP,
    /** Automatic hardware-GPU selection (native, no Python). */
    AUTO;

    public String token() {
        return name().toLowerCase(Locale.ROOT);
    }
}
