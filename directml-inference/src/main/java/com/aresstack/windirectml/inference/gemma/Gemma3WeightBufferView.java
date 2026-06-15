package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorEntry;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Decodes a SafeTensors tensor into an upload-ready FP32 buffer for the heap-light WARP weight load
 * (GEMMA-WARP-13a).
 *
 * <p>The product path uploads layer projections and the tied embedding/LM head from direct
 * little-endian FP32 {@link ByteBuffer}s instead of host {@code float[]} arrays, so the JVM heap no
 * longer carries the ~1 GB of weights. F16/BF16 tensors still need a one-off FP32 widen, but it lands in
 * an <b>off-heap direct</b> buffer (no million-entry {@code float[]}). The small norm vectors keep using
 * {@link Gemma3ReferenceWeights#decodeFloats} ({@code float[]}) — there is no benefit to buffering them.</p>
 */
public final class Gemma3WeightBufferView {

    private Gemma3WeightBufferView() {
    }

    /**
     * Decode {@code entry} (F32/F16/BF16) into a fresh <b>direct</b> little-endian FP32 {@link ByteBuffer},
     * positioned at 0 with {@code limit = elementCount * 4}. No host {@code float[]} is materialised.
     */
    public static ByteBuffer decodeFp32LittleEndian(SafeTensorEntry entry) throws IOException {
        long count = 1;
        for (long d : entry.shape()) {
            count = Math.multiplyExact(count, d);
        }
        int n = Math.toIntExact(count);
        ByteBuffer out = ByteBuffer.allocateDirect(Math.multiplyExact(n, Float.BYTES)).order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer src = entry.dataBuffer();
        switch (entry.dtype()) {
            case "F32" -> {
                for (int i = 0; i < n; i++) {
                    out.putFloat(src.getFloat());
                }
            }
            case "F16" -> {
                for (int i = 0; i < n; i++) {
                    out.putFloat(Float.float16ToFloat(src.getShort()));
                }
            }
            case "BF16" -> {
                for (int i = 0; i < n; i++) {
                    out.putFloat(Float.intBitsToFloat((src.getShort() & 0xFFFF) << 16));
                }
            }
            default -> throw new IOException("Unsupported Gemma weight dtype (expected F32/F16/BF16): " + entry.dtype());
        }
        out.flip();
        return out;
    }
}
