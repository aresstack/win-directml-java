package com.aresstack.windirectml.windows;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * COM interface IID constants and GUID memory layout helpers.
 * <p>
 * A COM GUID is 16 bytes: {@code {Data1: DWORD, Data2: WORD, Data3: WORD, Data4: BYTE[8]}}.
 * We store them as {@link MemorySegment}s in native memory so they can be
 * passed directly to Windows SDK functions that expect {@code REFIID}.
 * <p>
 * All IIDs here come straight from the Windows SDK headers.
 */
public final class ComIID {

    private ComIID() {}

    /** Standard GUID struct layout: 4 + 2 + 2 + 8 = 16 bytes. */
    public static final MemoryLayout GUID_LAYOUT = MemoryLayout.structLayout(
            ValueLayout.JAVA_INT.withName("Data1"),
            ValueLayout.JAVA_SHORT.withName("Data2"),
            ValueLayout.JAVA_SHORT.withName("Data3"),
            MemoryLayout.sequenceLayout(8, ValueLayout.JAVA_BYTE).withName("Data4")
    ).withName("GUID");

    /** Size of a GUID in bytes. */
    public static final long GUID_SIZE = GUID_LAYOUT.byteSize(); // 16

    // ── IUnknown ────────────────────────────────────────────────────────────
    /** IID_IUnknown = {00000000-0000-0000-C000-000000000046} */
    public static final byte[] IID_IUnknown_BYTES = guidBytes(
            0x00000000, (short) 0x0000, (short) 0x0000,
            (byte) 0xC0, (byte) 0x00, (byte) 0x00, (byte) 0x00,
            (byte) 0x00, (byte) 0x00, (byte) 0x00, (byte) 0x46);

    // ── DXGI ──────────────────────────────────────────────────────────────
    /** IID_IDXGIFactory1 = {770aae78-f26f-4dba-a829-253c83d1b387} */
    public static final byte[] IID_IDXGIFactory1_BYTES = guidBytes(
            0x770aae78, (short) 0xf26f, (short) 0x4dba,
            (byte) 0xa8, (byte) 0x29, (byte) 0x25, (byte) 0x3c,
            (byte) 0x83, (byte) 0xd1, (byte) 0xb3, (byte) 0x87);

    /** IID_IDXGIAdapter1 = {29038f61-3839-4626-91fd-086879011a05} */
    public static final byte[] IID_IDXGIAdapter1_BYTES = guidBytes(
            0x29038f61, (short) 0x3839, (short) 0x4626,
            (byte) 0x91, (byte) 0xfd, (byte) 0x08, (byte) 0x68,
            (byte) 0x79, (byte) 0x01, (byte) 0x1a, (byte) 0x05);

    // ── D3D12 ─────────────────────────────────────────────────────────────
    /** IID_ID3D12Device = {189819f1-1db6-4b57-be54-1821339b85f7} */
    public static final byte[] IID_ID3D12Device_BYTES = guidBytes(
            0x189819f1, (short) 0x1db6, (short) 0x4b57,
            (byte) 0xbe, (byte) 0x54, (byte) 0x18, (byte) 0x21,
            (byte) 0x33, (byte) 0x9b, (byte) 0x85, (byte) 0xf7);

    /** IID_ID3D12CommandQueue = {0ec870a6-5d7e-4c22-8cfc-5baae07616ed} */
    public static final byte[] IID_ID3D12CommandQueue_BYTES = guidBytes(
            0x0ec870a6, (short) 0x5d7e, (short) 0x4c22,
            (byte) 0x8c, (byte) 0xfc, (byte) 0x5b, (byte) 0xaa,
            (byte) 0xe0, (byte) 0x76, (byte) 0x16, (byte) 0xed);

    // ── DirectML ──────────────────────────────────────────────────────────
    /** IID_IDMLDevice = {6dbd6437-96fd-423f-a98c-ae5e7c2a573f} */
    public static final byte[] IID_IDMLDevice_BYTES = guidBytes(
            0x6dbd6437, (short) 0x96fd, (short) 0x423f,
            (byte) 0xa9, (byte) 0x8c, (byte) 0xae, (byte) 0x5e,
            (byte) 0x7c, (byte) 0x2a, (byte) 0x57, (byte) 0x3f);

    // ── D3D12 extended ────────────────────────────────────────────────────
    /** IID_ID3D12CommandAllocator = {6102dee4-af59-4b09-b999-b44d73f09b24} */
    public static final byte[] IID_ID3D12CommandAllocator_BYTES = guidBytes(
            0x6102dee4, (short) 0xaf59, (short) 0x4b09,
            (byte) 0xb9, (byte) 0x99, (byte) 0xb4, (byte) 0x4d,
            (byte) 0x73, (byte) 0xf0, (byte) 0x9b, (byte) 0x24);

    /** IID_ID3D12GraphicsCommandList = {5b160d0f-ac1b-4185-8ba8-b3ae42a5a455} */
    public static final byte[] IID_ID3D12GraphicsCommandList_BYTES = guidBytes(
            0x5b160d0f, (short) 0xac1b, (short) 0x4185,
            (byte) 0x8b, (byte) 0xa8, (byte) 0xb3, (byte) 0xae,
            (byte) 0x42, (byte) 0xa5, (byte) 0xa4, (byte) 0x55);

    /** IID_ID3D12Fence = {0a753dcf-c4d8-4b91-adf6-be5a60d95a76} */
    public static final byte[] IID_ID3D12Fence_BYTES = guidBytes(
            0x0a753dcf, (short) 0xc4d8, (short) 0x4b91,
            (byte) 0xad, (byte) 0xf6, (byte) 0xbe, (byte) 0x5a,
            (byte) 0x60, (byte) 0xd9, (byte) 0x5a, (byte) 0x76);

    /** IID_ID3D12DescriptorHeap = {8efb471d-616c-4f49-90f7-127bb763fa51} */
    public static final byte[] IID_ID3D12DescriptorHeap_BYTES = guidBytes(
            0x8efb471d, (short) 0x616c, (short) 0x4f49,
            (byte) 0x90, (byte) 0xf7, (byte) 0x12, (byte) 0x7b,
            (byte) 0xb7, (byte) 0x63, (byte) 0xfa, (byte) 0x51);

    /** IID_ID3D12Resource = {696442be-a72e-4059-bc79-5b5c98040fad} */
    public static final byte[] IID_ID3D12Resource_BYTES = guidBytes(
            0x696442be, (short) 0xa72e, (short) 0x4059,
            (byte) 0xbc, (byte) 0x79, (byte) 0x5b, (byte) 0x5c,
            (byte) 0x98, (byte) 0x04, (byte) 0x0f, (byte) 0xad);

    /** IID_ID3D12RootSignature = {c54a6b66-72df-4ee8-8be5-a946a1429214} */
    public static final byte[] IID_ID3D12RootSignature_BYTES = guidBytes(
            0xc54a6b66, (short) 0x72df, (short) 0x4ee8,
            (byte) 0x8b, (byte) 0xe5, (byte) 0xa9, (byte) 0x46,
            (byte) 0xa1, (byte) 0x42, (byte) 0x92, (byte) 0x14);

    /** IID_ID3D12PipelineState = {765a30f3-f624-4c6f-a828-ace948622445} */
    public static final byte[] IID_ID3D12PipelineState_BYTES = guidBytes(
            0x765a30f3, (short) 0xf624, (short) 0x4c6f,
            (byte) 0xa8, (byte) 0x28, (byte) 0xac, (byte) 0xe9,
            (byte) 0x48, (byte) 0x62, (byte) 0x24, (byte) 0x45);

    // ── DirectML extended ─────────────────────────────────────────────────
    /** IID_IDMLOperator = {26caae7a-3081-4633-9581-226fbe57695d} */
    public static final byte[] IID_IDMLOperator_BYTES = guidBytes(
            0x26caae7a, (short) 0x3081, (short) 0x4633,
            (byte) 0x95, (byte) 0x81, (byte) 0x22, (byte) 0x6f,
            (byte) 0xbe, (byte) 0x57, (byte) 0x69, (byte) 0x5d);

    /** IID_IDMLCompiledOperator = {6b15e56a-bf5c-4902-92d8-da3a650afea4} */
    public static final byte[] IID_IDMLCompiledOperator_BYTES = guidBytes(
            0x6b15e56a, (short) 0xbf5c, (short) 0x4902,
            (byte) 0x92, (byte) 0xd8, (byte) 0xda, (byte) 0x3a,
            (byte) 0x65, (byte) 0x0a, (byte) 0xfe, (byte) 0xa4);

    /** IID_IDMLOperatorInitializer = {427c1113-435c-469c-8676-4d5dd072f813} */
    public static final byte[] IID_IDMLOperatorInitializer_BYTES = guidBytes(
            0x427c1113, (short) 0x435c, (short) 0x469c,
            (byte) 0x86, (byte) 0x76, (byte) 0x4d, (byte) 0x5d,
            (byte) 0xd0, (byte) 0x72, (byte) 0xf8, (byte) 0x13);

    /** IID_IDMLCommandRecorder = {e6857a76-2e3e-4fdd-bff4-5d2ba10fb453} */
    public static final byte[] IID_IDMLCommandRecorder_BYTES = guidBytes(
            0xe6857a76, (short) 0x2e3e, (short) 0x4fdd,
            (byte) 0xbf, (byte) 0xf4, (byte) 0x5d, (byte) 0x2b,
            (byte) 0xa1, (byte) 0x0f, (byte) 0xb4, (byte) 0x53);

    /** IID_IDMLBindingTable = {29c687dc-de74-4e3b-ab00-1168f2fc3cfc} */
    public static final byte[] IID_IDMLBindingTable_BYTES = guidBytes(
            0x29c687dc, (short) 0xde74, (short) 0x4e3b,
            (byte) 0xab, (byte) 0x00, (byte) 0x11, (byte) 0x68,
            (byte) 0xf2, (byte) 0xfc, (byte) 0x3c, (byte) 0xfc);

    /**
     * Allocate a native GUID segment from the given byte representation.
     *
     * @param arena arena for the allocation
     * @param guid  16-byte GUID in little-endian SDK layout
     * @return a {@link MemorySegment} of size 16 that can be passed as {@code REFIID}
     */
    public static MemorySegment allocateGuid(Arena arena, byte[] guid) {
        MemorySegment seg = arena.allocate(GUID_LAYOUT);
        MemorySegment.copy(guid, 0, seg, ValueLayout.JAVA_BYTE, 0, 16);
        return seg;
    }

    // ── internal ──────────────────────────────────────────────────────────

    /**
     * Pack a GUID into a 16-byte array in the native struct layout
     * (little-endian Data1/Data2/Data3, then raw Data4 bytes).
     */
    private static byte[] guidBytes(int d1, short d2, short d3,
                                     byte b0, byte b1, byte b2, byte b3,
                                     byte b4, byte b5, byte b6, byte b7) {
        byte[] b = new byte[16];
        // Data1 – little-endian DWORD
        b[0] = (byte) (d1);
        b[1] = (byte) (d1 >>> 8);
        b[2] = (byte) (d1 >>> 16);
        b[3] = (byte) (d1 >>> 24);
        // Data2 – little-endian WORD
        b[4] = (byte) (d2);
        b[5] = (byte) (d2 >>> 8);
        // Data3 – little-endian WORD
        b[6] = (byte) (d3);
        b[7] = (byte) (d3 >>> 8);
        // Data4 – raw bytes
        b[8] = b0; b[9] = b1; b[10] = b2; b[11] = b3;
        b[12] = b4; b[13] = b5; b[14] = b6; b[15] = b7;
        return b;
    }
}

