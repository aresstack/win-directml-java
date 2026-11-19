package com.aresstack.audio.pipeline;

/**
 * Declare the technical requirements and nature of a block, so the pipeline and the editor can validate
 * a profile (supported sample rates, needed framing/history/look-ahead, whether it modifies the audio or
 * only produces metadata, whether it needs a learned noise/room profile) without a per-type switch.
 */
public interface AudioBlockCapabilities {

    boolean requiresFraming();

    int getPreferredFrameSize();

    int getRequiredLookAhead();

    int getRequiredHistory();

    /** @return the sample rates the block supports, or an empty array when it supports any rate. */
    int[] getSupportedSampleRates();

    boolean modifiesAudio();

    boolean producesMetadata();

    boolean requiresNoiseProfile();

    boolean requiresRoomProfile();

    boolean supportsStreaming();

    boolean supportsOffline();

    /**
     * @return whether the block leaves the channel count unchanged. Default true (most blocks preserve it;
     * format-changing blocks such as the channel mixer would return false). Declared as a default method so
     * existing {@link AudioBlockCapabilities} implementations stay source- and binary-compatible.
     */
    default boolean preservesChannelCount() {
        return true;
    }

    /** @return whether the block reads speech-activity metadata produced upstream (optional consumer). */
    default boolean consumesSpeechMetadata() {
        return false;
    }

    /** @return whether the block requires an upstream speech-activity track to function at all. */
    default boolean requiresSpeechActivityTrack() {
        return false;
    }

    /** @return whether the block changes the signal duration (and therefore the time base of later blocks). */
    default boolean changesDuration() {
        return false;
    }

    /** @return whether the block changes the sample count. */
    default boolean changesSampleCount() {
        return false;
    }

    /** @return whether the block needs the complete signal at once (i.e. is inherently offline, not streaming). */
    default boolean requiresCompleteSignal() {
        return false;
    }

    /** @return whether the block requires synchronized input channels (e.g. a beamformer). */
    default boolean requiresSynchronizedChannels() {
        return false;
    }

    /** @return whether the block requires a known microphone-array geometry (e.g. a beamformer). */
    default boolean requiresKnownMicrophoneGeometry() {
        return false;
    }
}
