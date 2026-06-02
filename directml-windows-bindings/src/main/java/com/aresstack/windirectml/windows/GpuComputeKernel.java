package com.aresstack.windirectml.windows;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

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

        // ── 1. Compile HLSL → DXBC ───────────────────────────────────
        MemorySegment shaderBlob = D3DCompilerBindings.compileShader(hlslSource, "CSMain", "cs_5_0", arena);
        MemorySegment shaderPtr = D3DCompilerBindings.blobGetBufferPointer(shaderBlob);
        long shaderSize = D3DCompilerBindings.blobGetBufferSize(shaderBlob);
        log.debug("Compiled compute shader '{}': {} bytes", name, shaderSize);

        // ── 2. Serialize + create root signature ──────────────────────
        MemorySegment rootSigBlob = D3DCompilerBindings.serializeRootSignature(numUavs, num32BitConsts, arena);
        MemorySegment rootSigPtr = D3DCompilerBindings.blobGetBufferPointer(rootSigBlob);
        long rootSigSize = D3DCompilerBindings.blobGetBufferSize(rootSigBlob);

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

        // Release shader blobs (bytecode is copied into PSO)
        DxgiBindings.release(shaderBlob);
        DxgiBindings.release(rootSigBlob);

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
        log.debug("GpuComputeKernel '{}' closed", name);
    }
}

