package com.aresstack.windirectml.windows;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Generic D3D12 compute shader kernel.
 * <p>
 * Wraps a compiled HLSL compute shader with its root signature and PSO.
 * Can record dispatches into a {@link GpuPipeline}'s command list.
 * <p>
 * Used for GPU-resident element-wise operations: add, RMSNorm, SwiGLU, scale.
 */
public final class GpuComputeKernel implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(GpuComputeKernel.class);

    /**
     * D3DCompile is expensive on WARP startup and many Qwen INT4 kernels use
     * the exact same HLSL source. Cache compiled DXBC bytes process-wide; each
     * GpuComputeKernel still creates its own root signature and PSO for now.
     */
    private static final ConcurrentMap<ShaderKey, byte[]> SHADER_BYTECODE_CACHE = new ConcurrentHashMap<>();
    private static final ConcurrentMap<RootSignatureKey, byte[]> ROOT_SIGNATURE_BYTECODE_CACHE = new ConcurrentHashMap<>();

    private record ShaderKey(String source, String entryPoint, String target) {
    }

    private record RootSignatureKey(int numUavs, int num32BitConsts) {
    }

    private final String name;
    private final Arena arena;
    private final int groupSizeX;

    // ── D3D12 objects ─────────────────────────────────────────────────
    private MemorySegment rootSignature;
    private MemorySegment pipelineState;

    // ── Pre-cached MethodHandles ──────────────────────────────────────
    private MethodHandle mhSetPipelineState;
    private MethodHandle mhSetComputeRootSig;
    private MethodHandle mhSetComputeRootUAV;
    private MethodHandle mhSetComputeRoot32Bit;
    private MethodHandle mhDispatch;

    private boolean closed = false;

    // ── ID3D12GraphicsCommandList vtable slots ──────────────────────────
    // Verified against known-working slots in D3D12Bindings:
    //   9=Close, 10=Reset, 15=CopyBufferRegion, 26=ResourceBarrier, 28=SetDescriptorHeaps
    private static final int CMDLIST_DISPATCH = 14;
    private static final int CMDLIST_SET_PIPELINE_STATE = 25;
    private static final int CMDLIST_SET_COMPUTE_ROOT_SIG = 29;
    private static final int CMDLIST_SET_COMPUTE_ROOT_32BIT = 35;  // SetComputeRoot32BitConstants (plural)
    private static final int CMDLIST_SET_COMPUTE_ROOT_UAV = 41;    // SetComputeRootUnorderedAccessView

    // ── ID3D12Device vtable slots ────────────────────────────────────────
    // Verified against D3D12Bindings: 8=CreateCommandQueue, 9=CreateCommandAllocator,
    //   12=CreateCommandList, 14=CreateDescriptorHeap, 27=CreateCommittedResource, 36=CreateFence
    private static final int DEV_CREATE_COMPUTE_PSO = 11;
    private static final int DEV_CREATE_ROOT_SIGNATURE = 16;

    /**
     * Create a compute kernel from HLSL source code.
     *
     * @param wb             initialized WindowsBindings
     * @param cmdListForMH   a command list for caching MethodHandles (shared pipeline's)
     * @param hlslSource     HLSL source code
     * @param name           kernel name (for logging)
     * @param numUavs        number of UAV root descriptors
     * @param num32BitConsts number of root 32-bit constants
     * @param groupSizeX     thread group size (matches [numthreads(X,1,1)] in shader)
     */
    public GpuComputeKernel(WindowsBindings wb, MemorySegment cmdListForMH,
                            String hlslSource, String name,
                            int numUavs, int num32BitConsts, int groupSizeX)
            throws WindowsNativeException {
        this.name = name;
        this.arena = Arena.ofShared();
        this.groupSizeX = groupSizeX;

        var dev = wb.getD3d12Device();

        // ── 1. Compile HLSL → DXBC (cached across equal sources) ───────
        byte[] shaderBytecode = cachedShaderBytecode(hlslSource, "CSMain", "cs_5_0", name);
        MemorySegment shaderData = arena.allocate(shaderBytecode.length, 8);
        shaderData.copyFrom(MemorySegment.ofArray(shaderBytecode));
        MemorySegment shaderPtr = shaderData;
        long shaderSize = shaderBytecode.length;
        log.debug("Using compute shader bytecode '{}': {} bytes", name, shaderSize);

        // ── 2. Serialize + create root signature (serialized bytes cached) ─
        byte[] rootSigBytecode = cachedRootSignatureBytecode(numUavs, num32BitConsts, name);
        MemorySegment rootSigData = arena.allocate(rootSigBytecode.length, 8);
        rootSigData.copyFrom(MemorySegment.ofArray(rootSigBytecode));
        MemorySegment rootSigPtr = rootSigData;
        long rootSigSize = rootSigBytecode.length;

        try {
            MemorySegment riid = ComIID.allocateGuid(arena, ComIID.IID_ID3D12RootSignature_BYTES);
            MemorySegment pp = arena.allocate(ValueLayout.ADDRESS);
            MethodHandle mh = DxgiBindings.vtableMethod(dev, DEV_CREATE_ROOT_SIGNATURE,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            int hr = (int) mh.invokeExact(dev, 0, rootSigPtr, rootSigSize, riid, pp);
            HResult.check(hr, "CreateRootSignature(" + name + ")");
            rootSignature = pp.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("CreateRootSignature failed", t);
        }

        // ── 3. Create compute PSO ─────────────────────────────────────
        try {
            // D3D12_COMPUTE_PIPELINE_STATE_DESC:
            // pRootSignature(8) + CS.pShaderBytecode(8) + CS.BytecodeLength(8)
            // + NodeMask(4) + CachedPSO.pCachedBlob(8) + CachedPSO.CachedBlobSizeInBytes(8) + Flags(4)
            // = 48 bytes (with alignment)
            MemorySegment psoDesc = arena.allocate(56, 8);
            psoDesc.set(ValueLayout.ADDRESS, 0, rootSignature);
            psoDesc.set(ValueLayout.ADDRESS, 8, shaderPtr);      // CS.pShaderBytecode
            psoDesc.set(ValueLayout.JAVA_LONG, 16, shaderSize);  // CS.BytecodeLength
            psoDesc.set(ValueLayout.JAVA_INT, 24, 0);            // NodeMask
            psoDesc.set(ValueLayout.ADDRESS, 32, MemorySegment.NULL); // CachedPSO.pCachedBlob
            psoDesc.set(ValueLayout.JAVA_LONG, 40, 0L);          // CachedPSO size
            psoDesc.set(ValueLayout.JAVA_INT, 48, 0);            // Flags

            MemorySegment riid = ComIID.allocateGuid(arena, ComIID.IID_ID3D12PipelineState_BYTES);
            MemorySegment pp = arena.allocate(ValueLayout.ADDRESS);
            MethodHandle mh = DxgiBindings.vtableMethod(dev, DEV_CREATE_COMPUTE_PSO,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            int hr = (int) mh.invokeExact(dev, psoDesc, riid, pp);
            HResult.check(hr, "CreateComputePipelineState(" + name + ")");
            pipelineState = pp.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("CreateComputePSO failed", t);
        }

        // ── 4. Pre-cache MethodHandles for the command list ───────────
        mhSetPipelineState = DxgiBindings.vtableMethod(cmdListForMH, CMDLIST_SET_PIPELINE_STATE,
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        mhSetComputeRootSig = DxgiBindings.vtableMethod(cmdListForMH, CMDLIST_SET_COMPUTE_ROOT_SIG,
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        mhSetComputeRootUAV = DxgiBindings.vtableMethod(cmdListForMH, CMDLIST_SET_COMPUTE_ROOT_UAV,
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_LONG));
        mhSetComputeRoot32Bit = DxgiBindings.vtableMethod(cmdListForMH, CMDLIST_SET_COMPUTE_ROOT_32BIT,
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT,    // RootParameterIndex
                        ValueLayout.JAVA_INT,    // Num32BitValuesToSet
                        ValueLayout.ADDRESS,     // pSrcData
                        ValueLayout.JAVA_INT));  // DestOffsetIn32BitValues
        mhDispatch = DxgiBindings.vtableMethod(cmdListForMH, CMDLIST_DISPATCH,
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));

        log.info("GpuComputeKernel '{}' ready (groupSize={})", name, groupSizeX);
    }

    private static byte[] cachedShaderBytecode(String hlslSource, String entryPoint,
                                               String target, String kernelName)
            throws WindowsNativeException {
        ShaderKey key = new ShaderKey(hlslSource, entryPoint, target);
        byte[] cached = SHADER_BYTECODE_CACHE.get(key);
        if (cached != null) {
            return cached;
        }
        synchronized (SHADER_BYTECODE_CACHE) {
            cached = SHADER_BYTECODE_CACHE.get(key);
            if (cached != null) {
                return cached;
            }
            byte[] compiled = compileShaderBytecode(hlslSource, entryPoint, target);
            SHADER_BYTECODE_CACHE.put(key, compiled);
            log.info("Compiled compute shader bytecode '{}' once: {} bytes (cacheSize={})",
                    kernelName, compiled.length, SHADER_BYTECODE_CACHE.size());
            return compiled;
        }
    }

    private static byte[] cachedRootSignatureBytecode(int numUavs, int num32BitConsts,
                                                      String kernelName)
            throws WindowsNativeException {
        RootSignatureKey key = new RootSignatureKey(numUavs, num32BitConsts);
        byte[] cached = ROOT_SIGNATURE_BYTECODE_CACHE.get(key);
        if (cached != null) {
            return cached;
        }
        synchronized (ROOT_SIGNATURE_BYTECODE_CACHE) {
            cached = ROOT_SIGNATURE_BYTECODE_CACHE.get(key);
            if (cached != null) {
                return cached;
            }
            byte[] serialized = serializeRootSignatureBytecode(numUavs, num32BitConsts);
            ROOT_SIGNATURE_BYTECODE_CACHE.put(key, serialized);
            log.info("Serialized root signature for '{}' once: uavs={}, constants={}, {} bytes (cacheSize={})",
                    kernelName, numUavs, num32BitConsts, serialized.length,
                    ROOT_SIGNATURE_BYTECODE_CACHE.size());
            return serialized;
        }
    }

    private static byte[] compileShaderBytecode(String hlslSource, String entryPoint,
                                                String target) throws WindowsNativeException {
        MemorySegment blob = MemorySegment.NULL;
        try (Arena temp = Arena.ofConfined()) {
            blob = D3DCompilerBindings.compileShader(hlslSource, entryPoint, target, temp);
            return copyBlobToByteArray(blob);
        } finally {
            if (!blob.equals(MemorySegment.NULL)) {
                DxgiBindings.release(blob);
            }
        }
    }

    private static byte[] serializeRootSignatureBytecode(int numUavs, int num32BitConsts)
            throws WindowsNativeException {
        MemorySegment blob = MemorySegment.NULL;
        try (Arena temp = Arena.ofConfined()) {
            blob = D3DCompilerBindings.serializeRootSignature(numUavs, num32BitConsts, temp);
            return copyBlobToByteArray(blob);
        } finally {
            if (!blob.equals(MemorySegment.NULL)) {
                DxgiBindings.release(blob);
            }
        }
    }

    private static byte[] copyBlobToByteArray(MemorySegment blob) {
        MemorySegment ptr = D3DCompilerBindings.blobGetBufferPointer(blob);
        int size = Math.toIntExact(D3DCompilerBindings.blobGetBufferSize(blob));
        byte[] bytes = new byte[size];
        MemorySegment.ofArray(bytes).copyFrom(ptr.reinterpret(size));
        return bytes;
    }

    /**
     * Record this compute shader dispatch into a pipeline's command list.
     *
     * @param cl           the command list (from GpuPipeline)
     * @param uavAddresses GPU virtual addresses for each UAV parameter
     * @param constants    32-bit constant values (can be null if no constants)
     * @param elementCount total elements to process (dispatch groups = ceil(count / groupSize))
     */
    public void recordDispatch(MemorySegment cl, long[] uavAddresses,
                               int[] constants, int elementCount) {
        try {
            // Set PSO + root signature
            mhSetPipelineState.invokeExact(cl, pipelineState);
            mhSetComputeRootSig.invokeExact(cl, rootSignature);

            // Set UAV root descriptors
            for (int i = 0; i < uavAddresses.length; i++) {
                mhSetComputeRootUAV.invokeExact(cl, i, uavAddresses[i]);
            }

            // Set root constants
            if (constants != null && constants.length > 0) {
                MemorySegment constBuf = arena.allocate((long) constants.length * 4, 4);
                for (int i = 0; i < constants.length; i++) {
                    constBuf.set(ValueLayout.JAVA_INT, (long) i * 4, constants[i]);
                }
                int rootIdx = uavAddresses.length; // constants param is after UAVs
                mhSetComputeRoot32Bit.invokeExact(cl, rootIdx, constants.length, constBuf, 0);
            }

            // Dispatch
            int groups = (elementCount + groupSizeX - 1) / groupSizeX;
            mhDispatch.invokeExact(cl, groups, 1, 1);
            WarpSubmissionStats.recordDispatch(); // GEMMA-WARP-14b: count recorded dispatches

        } catch (Throwable t) {
            throw new RuntimeException("GpuComputeKernel.recordDispatch(" + name + ") failed", t);
        }
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        if (pipelineState != null) DxgiBindings.release(pipelineState);
        if (rootSignature != null) DxgiBindings.release(rootSignature);
        arena.close();
        log.trace("GpuComputeKernel '{}' closed", name);
    }
}

