package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyKvCacheLayout;
import com.aresstack.windirectml.windows.D3D12Bindings;
import com.aresstack.windirectml.windows.DxgiBindings;
import com.aresstack.windirectml.windows.WindowsBindings;
import com.aresstack.windirectml.windows.WindowsNativeException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * GPU-resident KV cache for Qwen2 decode (Opt-B).
 *
 * <p>One D3D12 DEFAULT-heap buffer per (layer, K/V) pair, layout
 * {@code [kvHeads * maxSeqLen * headDim]} float32 row-major
 * (slowest-changing axis = kvHead, then position, then dim — matches
 * CPU layout {@code kvCacheK[layer][kvH][pos*headDim+d]}).
 *
 * <p>Memory budget for Qwen2.5-Coder-0.5B (kvHeads=2, headDim=64, maxSeqLen=4096):
 * <pre>
 *   per layer K: 2 * 4096 * 64 * 4 B  ≈   2.0 MB
 *   per layer V: 2 * 4096 * 64 * 4 B  ≈   2.0 MB
 *   × 24 layers × 2 buffers           ≈  96  MB on GPU
 * </pre>
 *
 * <p>Used by:
 * <ul>
 *   <li>{@code QwenAttentionShaders.rope_and_append} — writes rotated K and V
 *       at position {@code pos} during decode.</li>
 *   <li>{@code QwenAttentionShaders.gqa_attention_decode} — reads K[0..pos]
 *       and V[0..pos] per kvHead during decode attention.</li>
 *   <li>{@link #uploadFromCpu} — uploads the CPU-built prefill KV cache so
 *       decode starts with the correct prompt context (B-Step 6).</li>
 * </ul>
 *
 * <p>Thread safety: not thread-safe. Caller must ensure single-threaded access
 * (decode loop is single-threaded by design).
 *
 * @see QwenAttentionShaders
 */
public final class QwenGpuKvCache implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(QwenGpuKvCache.class);

    private final WindowsBindings wb;
    private final int numLayers;
    private final int kvHeads;
    private final int headDim;
    private final int maxSeqLen;
    private final DecoderOnlyKvCacheLayout layout;
    private final long bytesPerLayer;
    private final Arena arena;

    /**
     * Per-layer GPU buffers: {@code kBuf[layer]} holds K, {@code vBuf[layer]} holds V.
     */
    private final MemorySegment[] kBuf;
    private final MemorySegment[] vBuf;

    /**
     * Cached GPU virtual addresses (avoid vtable call on each dispatch).
     */
    private final long[] kAddr;
    private final long[] vAddr;

    private boolean closed = false;

    /**
     * Allocate GPU buffers for the KV cache.
     *
     * @param wb        initialised WindowsBindings (D3D12 device required)
     * @param numLayers number of decoder layers
     * @param kvHeads   number of KV heads per layer
     * @param headDim   dimension per attention head
     * @param maxSeqLen maximum sequence length (must accommodate prompt + max decode)
     */
    public QwenGpuKvCache(WindowsBindings wb, int numLayers, int kvHeads,
                          int headDim, int maxSeqLen) {
        this.wb = wb;
        this.numLayers = numLayers;
        this.kvHeads = kvHeads;
        this.headDim = headDim;
        this.maxSeqLen = maxSeqLen;
        this.layout = new DecoderOnlyKvCacheLayout(numLayers, kvHeads, maxSeqLen, headDim);
        this.bytesPerLayer = layout.bytesPerLayer();
        this.arena = Arena.ofShared();
        this.kBuf = new MemorySegment[numLayers];
        this.vBuf = new MemorySegment[numLayers];
        this.kAddr = new long[numLayers];
        this.vAddr = new long[numLayers];

        try {
            var dev = wb.getD3d12Device();
            for (int l = 0; l < numLayers; l++) {
                kBuf[l] = D3D12Bindings.createDefaultBuffer(dev, bytesPerLayer, arena);
                vBuf[l] = D3D12Bindings.createDefaultBuffer(dev, bytesPerLayer, arena);
                kAddr[l] = D3D12Bindings.getGpuVirtualAddress(kBuf[l]);
                vAddr[l] = D3D12Bindings.getGpuVirtualAddress(vBuf[l]);
            }
        } catch (WindowsNativeException e) {
            arena.close();
            throw new RuntimeException("QwenGpuKvCache allocation failed", e);
        }

        long totalMb = 2L * numLayers * bytesPerLayer / (1024 * 1024);
        log.info("QwenGpuKvCache ready: {} layers × 2 (K+V) × {} KB = {} MB on GPU "
                        + "(kvHeads={}, maxSeqLen={}, headDim={})",
                numLayers, bytesPerLayer / 1024, totalMb,
                kvHeads, maxSeqLen, headDim);
    }

    public int getNumLayers() {
        return numLayers;
    }

    public int getKvHeads() {
        return kvHeads;
    }

    public int getHeadDim() {
        return headDim;
    }

    public int getMaxSeqLen() {
        return maxSeqLen;
    }

    public DecoderOnlyKvCacheLayout getLayout() {
        return layout;
    }

    /**
     * GPU virtual address of K buffer for the given layer.
     */
    public long getKAddr(int layer) {
        return kAddr[layer];
    }

    /**
     * GPU virtual address of V buffer for the given layer.
     */
    public long getVAddr(int layer) {
        return vAddr[layer];
    }

    /**
     * Raw GPU buffer handle for K (for advanced binding scenarios).
     */
    public MemorySegment getKBuf(int layer) {
        return kBuf[layer];
    }

    /**
     * Raw GPU buffer handle for V (for advanced binding scenarios).
     */
    public MemorySegment getVBuf(int layer) {
        return vBuf[layer];
    }

    /**
     * Upload prefill KV-cache contents from CPU arrays into the GPU buffer
     * for one layer (B-Step 6: call once at the end of prefill, before the
     * first decode token).
     *
     * <p>Source layout: {@code cpuK[kvHead][pos * headDim + d]} (matches
     * {@code Qwen2Runtime.kvCacheK[layer][kvHead]}).
     * Target layout: contiguous {@code [kvHeads * maxSeqLen * headDim]}
     * row-major (kvHead → pos → d).
     *
     * <p>Only the first {@code seqLen} positions per kvHead are written;
     * the remaining {@code maxSeqLen - seqLen} positions stay
     * uninitialized — decode must never read past the current position.
     *
     * @param layer  decoder layer index
     * @param seqLen number of valid positions to upload (must be ≤ maxSeqLen)
     * @param cpuK   CPU K cache: {@code float[kvHeads][>= seqLen * headDim]}
     * @param cpuV   CPU V cache: {@code float[kvHeads][>= seqLen * headDim]}
     */
    public void uploadFromCpu(int layer, int seqLen, float[][] cpuK, float[][] cpuV) {
        if (closed) throw new IllegalStateException("KV cache closed");
        if (seqLen <= 0 || seqLen > maxSeqLen) {
            throw new IllegalArgumentException(
                    "seqLen=" + seqLen + " out of [1.." + maxSeqLen + "]");
        }
        if (cpuK.length < kvHeads || cpuV.length < kvHeads) {
            throw new IllegalArgumentException(
                    "cpuK/cpuV must have at least " + kvHeads + " heads");
        }

        // Build one contiguous staging buffer per layer in target GPU layout
        // [kvHead -> pos -> d]. Tail positions (seqLen .. maxSeqLen-1) stay at 0;
        // the attention shader must clamp its loop at the current pos and never
        // read past it (decode passes pos+1 via root constant).
        int totalFloats = Math.toIntExact(layout.bytesPerLayer() / Float.BYTES);
        float[] stagingK = new float[totalFloats];
        float[] stagingV = new float[totalFloats];
        int validFloats = layout.validPrefixLength(seqLen);
        for (int kvH = 0; kvH < kvHeads; kvH++) {
            int dstOff = kvH * maxSeqLen * headDim;
            // cpuK[kvH] may be larger than seqLen*headDim (allocated for max
            // capacity); copy only the valid prefix.
            System.arraycopy(cpuK[kvH], 0, stagingK, dstOff, validFloats);
            System.arraycopy(cpuV[kvH], 0, stagingV, dstOff, validFloats);
        }

        try {
            var dev = wb.getD3d12Device();
            var queue = wb.getCommandQueue();
            D3D12Bindings.uploadFloats(dev, queue, kBuf[layer], stagingK, arena);
            D3D12Bindings.uploadFloats(dev, queue, vBuf[layer], stagingV, arena);
            if (layer == 0) {
                log.debug("KV cache upload layer={} seqLen={} ({} KB per K/V buffer, full stride)",
                        layer, seqLen, totalFloats * Float.BYTES / 1024);
            }
        } catch (WindowsNativeException e) {
            throw new RuntimeException(
                    "KV cache upload failed for layer " + layer, e);
        }
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        for (int l = 0; l < numLayers; l++) {
            if (kBuf[l] != null) DxgiBindings.release(kBuf[l]);
            if (vBuf[l] != null) DxgiBindings.release(vBuf[l]);
        }
        arena.close();
        log.info("QwenGpuKvCache closed ({} layers × 2 buffers released)", numLayers);
    }
}
