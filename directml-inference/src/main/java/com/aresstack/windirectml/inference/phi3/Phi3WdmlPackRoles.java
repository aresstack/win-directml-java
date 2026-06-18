package com.aresstack.windirectml.inference.phi3;

/**
 * Canonical tensor-role names written into / read from the Phi-3 {@code .wdmlpack}
 * (PHI3-WDMLPACK-COMPILER-1). Keeping the names in one place lets the compiler and the runtime-package loader agree
 * without re-deriving the fragile ONNX graph node names.
 *
 * <p>Quantized (INT4 MatMulNBits) projections are stored as a triplet: {@code <role>.qweight} (UINT8, dims
 * {@code [N, K, blockSize]}), {@code <role>.scales} (FLOAT) and {@code <role>.zeropoints} (UINT8). FP32 vectors are
 * stored as a single FLOAT tensor.</p>
 */
public final class Phi3WdmlPackRoles {

    public static final String EMBED_TOKENS = "embed_tokens";
    public static final String COS_CACHE = "cos_cache";
    public static final String SIN_CACHE = "sin_cache";
    public static final String FINAL_NORM = "final_norm";
    public static final String LM_HEAD = "lm_head";

    public static final String INPUT_NORM = "input_norm";
    public static final String POST_NORM = "post_norm";
    public static final String ATTN_OUT_SCALE = "attn_out_scale";
    public static final String MLP_OUT_SCALE = "mlp_out_scale";
    public static final String Q_PROJ = "q_proj";
    public static final String K_PROJ = "k_proj";
    public static final String V_PROJ = "v_proj";
    public static final String O_PROJ = "o_proj";
    public static final String GATE_UP_PROJ = "gate_up_proj";
    public static final String DOWN_PROJ = "down_proj";

    public static final String QWEIGHT_SUFFIX = ".qweight";
    public static final String SCALES_SUFFIX = ".scales";
    public static final String ZEROPOINTS_SUFFIX = ".zeropoints";

    /** ONNX-compatible element type codes used in the tensor directory. */
    public static final int DTYPE_FLOAT = 1;
    public static final int DTYPE_UINT8 = 2;

    private Phi3WdmlPackRoles() {
    }

    public static String layer(int layerIndex, String role) {
        return "layers." + layerIndex + "." + role;
    }

    public static String qweight(String role) {
        return role + QWEIGHT_SUFFIX;
    }

    public static String scales(String role) {
        return role + SCALES_SUFFIX;
    }

    public static String zeropoints(String role) {
        return role + ZEROPOINTS_SUFFIX;
    }
}
