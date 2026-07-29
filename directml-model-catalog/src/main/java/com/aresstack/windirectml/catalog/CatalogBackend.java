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
    /**
     * The DirectML code path bound to a specific DirectX-12 GPU (Phi-3's {@code -Dphi3.backend=directml}).
     * For the WARP/AUTO families, {@link #WARP} and {@link #AUTO} are the two adapter selections of this
     * same DirectML path, so they list those instead of a bare {@code DIRECTML}.
     */
    DIRECTML,
    /** The same DirectML code path bound to the D3D12 WARP <em>software</em> adapter (native, no Python). */
    WARP,
    /** The same DirectML code path bound to an automatic <em>hardware</em> GPU adapter (native, no Python). */
    AUTO;

    public String token() {
        return name().toLowerCase(Locale.ROOT);
    }
}
