package com.aresstack.windirectml.windows;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * FFM bindings for the D3D shader compiler ({@code d3dcompiler_47.dll}).
 * <p>
 * Provides runtime HLSL → DXBC compilation for custom compute shaders
 * (element-wise add, RMSNorm, SwiGLU) used by the GPU pipeline.
 */
public final class D3DCompilerBindings {

    private static final Logger log = LoggerFactory.getLogger(D3DCompilerBindings.class);

    private D3DCompilerBindings() {}

    // D3DCompile function from d3dcompiler_47.dll
    private static volatile MethodHandle d3dCompileHandle;

    private static MethodHandle getD3DCompileHandle() {
        if (d3dCompileHandle == null) {
            synchronized (D3DCompilerBindings.class) {
                if (d3dCompileHandle == null) {
                    SymbolLookup lib = SymbolLookup.libraryLookup("d3dcompiler_47.dll", Arena.global());
                    MemorySegment addr = lib.find("D3DCompile")
                            .orElseThrow(() -> new UnsatisfiedLinkError("D3DCompile not found in d3dcompiler_47.dll"));
                    d3dCompileHandle = Linker.nativeLinker().downcallHandle(addr,
                            FunctionDescriptor.of(
                                    ValueLayout.JAVA_INT,    // HRESULT
                                    ValueLayout.ADDRESS,     // pSrcData
                                    ValueLayout.JAVA_LONG,   // SrcDataSize
                                    ValueLayout.ADDRESS,     // pSourceName (nullable)
                                    ValueLayout.ADDRESS,     // pDefines (nullable)
                                    ValueLayout.ADDRESS,     // pInclude (nullable)
                                    ValueLayout.ADDRESS,     // pEntrypoint
                                    ValueLayout.ADDRESS,     // pTarget
                                    ValueLayout.JAVA_INT,    // Flags1
                                    ValueLayout.JAVA_INT,    // Flags2
                                    ValueLayout.ADDRESS,     // ppCode (ID3DBlob**)
                                    ValueLayout.ADDRESS      // ppErrorMsgs (ID3DBlob**)
                            ));
                    log.debug("Resolved D3DCompile at {}", addr);
                }
            }
        }
        return d3dCompileHandle;
    }

    // D3D12SerializeRootSignature from d3d12.dll
    private static volatile MethodHandle serializeRootSigHandle;

    private static MethodHandle getSerializeRootSigHandle() {
        if (serializeRootSigHandle == null) {
            synchronized (D3DCompilerBindings.class) {
                if (serializeRootSigHandle == null) {
                    SymbolLookup d3d12 = SymbolLookup.libraryLookup("d3d12.dll", Arena.global());
                    MemorySegment addr = d3d12.find("D3D12SerializeRootSignature")
                            .orElseThrow(() -> new UnsatisfiedLinkError("D3D12SerializeRootSignature not found"));
                    serializeRootSigHandle = Linker.nativeLinker().downcallHandle(addr,
                            FunctionDescriptor.of(
                                    ValueLayout.JAVA_INT,    // HRESULT
                                    ValueLayout.ADDRESS,     // pRootSignature (D3D12_ROOT_SIGNATURE_DESC*)
                                    ValueLayout.JAVA_INT,    // Version
                                    ValueLayout.ADDRESS,     // ppBlob (ID3DBlob**)
                                    ValueLayout.ADDRESS      // ppErrorBlob (ID3DBlob**)
                            ));
                }
            }
        }
        return serializeRootSigHandle;
    }

    // ═══════════════════════════════════════════════════════════════════
    // HLSL compilation
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Compile HLSL source code to DXBC bytecode.
     *
     * @param hlslSource shader source code
     * @param entryPoint entry function name (e.g., "CSMain")
     * @param target     shader model (e.g., "cs_5_0")
     * @param arena      arena for allocations
     * @return ID3DBlob containing compiled bytecode (caller must release)
     */
    public static MemorySegment compileShader(String hlslSource, String entryPoint,
                                               String target, Arena arena)
            throws WindowsNativeException {
        try {
            byte[] srcBytes = hlslSource.getBytes(java.nio.charset.StandardCharsets.UTF_8);
            MemorySegment srcData = arena.allocate(srcBytes.length);
            srcData.copyFrom(MemorySegment.ofArray(srcBytes));

            MemorySegment entry = arena.allocateUtf8String(entryPoint);
            MemorySegment tgt = arena.allocateUtf8String(target);
            MemorySegment ppCode = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment ppErrors = arena.allocate(ValueLayout.ADDRESS);

            int hr = (int) getD3DCompileHandle().invokeExact(
                    srcData, (long) srcBytes.length,
                    MemorySegment.NULL, MemorySegment.NULL, MemorySegment.NULL,
                    entry, tgt, 0, 0, ppCode, ppErrors);

            if (hr != 0) {
                MemorySegment errBlob = ppErrors.get(ValueLayout.ADDRESS, 0);
                String errMsg = "[no error message]";
                if (!errBlob.equals(MemorySegment.NULL)) {
                    errBlob = errBlob.reinterpret(Long.MAX_VALUE);
                    errMsg = readBlobString(errBlob);
                    DxgiBindings.release(errBlob);
                }
                throw new WindowsNativeException("D3DCompile failed (0x" +
                        Integer.toHexString(hr) + "): " + errMsg);
            }

            return ppCode.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
        } catch (WindowsNativeException e) { throw e; }
        catch (Throwable t) { throw new WindowsNativeException("D3DCompile invocation failed", t); }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Root signature serialization
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Serialize a root signature description to a blob.
     * <p>
     * Creates a root signature with N root UAV descriptors + M root 32-bit constants.
     *
     * @param numUavs       number of UAV root descriptors (u0, u1, ...)
     * @param num32BitConsts number of 32-bit constants in the root constant buffer (b0)
     * @param arena         arena
     * @return ID3DBlob with serialized root signature
     */
    public static MemorySegment serializeRootSignature(int numUavs, int num32BitConsts,
                                                        Arena arena)
            throws WindowsNativeException {
        try {
            // D3D12_ROOT_PARAMETER layout (x64): 32 bytes
            // Type(4) + union(24) + ShaderVisibility(4)
            int totalParams = numUavs + (num32BitConsts > 0 ? 1 : 0);
            MemorySegment params = arena.allocate(32L * totalParams, 8);

            for (int i = 0; i < numUavs; i++) {
                long off = (long) i * 32;
                params.set(ValueLayout.JAVA_INT, off, 4); // D3D12_ROOT_PARAMETER_TYPE_UAV = 4
                // D3D12_ROOT_DESCRIPTOR: ShaderRegister(4) + RegisterSpace(4)
                params.set(ValueLayout.JAVA_INT, off + 8, i);  // register(uN)
                params.set(ValueLayout.JAVA_INT, off + 12, 0); // space0
                params.set(ValueLayout.JAVA_INT, off + 24, 0); // ALL visibility
            }

            if (num32BitConsts > 0) {
                long off = (long) numUavs * 32;
                params.set(ValueLayout.JAVA_INT, off, 1); // D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS = 1
                // D3D12_ROOT_CONSTANTS: ShaderRegister(4) + RegisterSpace(4) + Num32BitValues(4)
                params.set(ValueLayout.JAVA_INT, off + 8, 0);  // b0
                params.set(ValueLayout.JAVA_INT, off + 12, 0); // space0
                params.set(ValueLayout.JAVA_INT, off + 16, num32BitConsts);
                params.set(ValueLayout.JAVA_INT, off + 24, 0); // ALL visibility
            }

            // D3D12_ROOT_SIGNATURE_DESC: NumParameters(4)+pad(4)+pParameters(8)+NumStaticSamplers(4)+pad(4)+pStaticSamplers(8)+Flags(4)+pad(4)
            MemorySegment desc = arena.allocate(40, 8);
            desc.set(ValueLayout.JAVA_INT, 0, totalParams);
            desc.set(ValueLayout.ADDRESS, 8, params);
            desc.set(ValueLayout.JAVA_INT, 16, 0); // no static samplers
            desc.set(ValueLayout.ADDRESS, 24, MemorySegment.NULL);
            desc.set(ValueLayout.JAVA_INT, 32, 0); // D3D12_ROOT_SIGNATURE_FLAG_NONE

            MemorySegment ppBlob = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment ppErr = arena.allocate(ValueLayout.ADDRESS);

            int hr = (int) getSerializeRootSigHandle().invokeExact(
                    desc,
                    1, // D3D_ROOT_SIGNATURE_VERSION_1
                    ppBlob, ppErr);

            if (hr != 0) {
                MemorySegment errBlob = ppErr.get(ValueLayout.ADDRESS, 0);
                String errMsg = "[no error message]";
                if (!errBlob.equals(MemorySegment.NULL)) {
                    errBlob = errBlob.reinterpret(Long.MAX_VALUE);
                    errMsg = readBlobString(errBlob);
                    DxgiBindings.release(errBlob);
                }
                throw new WindowsNativeException("SerializeRootSignature failed: " + errMsg);
            }

            return ppBlob.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
        } catch (WindowsNativeException e) { throw e; }
        catch (Throwable t) { throw new WindowsNativeException("SerializeRootSignature failed", t); }
    }

    // ═══════════════════════════════════════════════════════════════════
    // ID3DBlob helpers
    // ═══════════════════════════════════════════════════════════════════

    /** ID3DBlob::GetBufferPointer (vtable slot 3). */
    public static MemorySegment blobGetBufferPointer(MemorySegment blob) {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(blob, 3,
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            MemorySegment ptr = (MemorySegment) mh.invokeExact(blob);
            return ptr.reinterpret(Long.MAX_VALUE);
        } catch (Throwable t) { throw new RuntimeException("Blob::GetBufferPointer failed", t); }
    }

    /** ID3DBlob::GetBufferSize (vtable slot 4). */
    public static long blobGetBufferSize(MemorySegment blob) {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(blob, 4,
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));
            return (long) mh.invokeExact(blob);
        } catch (Throwable t) { throw new RuntimeException("Blob::GetBufferSize failed", t); }
    }

    /** Read a blob's content as a UTF-8 string (for error messages). */
    private static String readBlobString(MemorySegment blob) {
        MemorySegment ptr = blobGetBufferPointer(blob);
        long size = blobGetBufferSize(blob);
        byte[] bytes = new byte[(int) Math.min(size, 4096)];
        MemorySegment.copy(ptr, ValueLayout.JAVA_BYTE, 0, bytes, 0, bytes.length);
        return new String(bytes, java.nio.charset.StandardCharsets.UTF_8).trim();
    }
}

