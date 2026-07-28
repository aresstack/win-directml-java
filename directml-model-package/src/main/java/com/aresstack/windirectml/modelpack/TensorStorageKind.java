package com.aresstack.windirectml.modelpack;

/**
 * Describes where a tensor payload lives before runtime upload.
 */
public enum TensorStorageKind {
    /** Payload bytes are embedded directly in the source file. */
    INLINE,
    /** Payload bytes live in a separate external data file. */
    EXTERNAL,
    /** Tensor metadata exists but the payload is absent or intentionally skipped. */
    METADATA_ONLY
}
