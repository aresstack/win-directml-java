package com.aresstack.windirectml.inference.artifact;

/**
 * State of the raw, downloaded model source files (SafeTensors / ONNX / checkpoint),
 * independent of any compiled runtime package.
 */
public enum RawAssetState {
    /** The model directory does not exist. */
    RAW_MISSING,
    /** The directory exists but a required source file is absent (interrupted/partial download). */
    RAW_INCOMPLETE,
    /** A required source file exists but is zero bytes (truncated download). */
    RAW_CORRUPT,
    /** All required source files are present and non-empty. */
    RAW_VALID
}
