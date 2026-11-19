package com.aresstack.audio.pipeline;

/**
 * Immutable {@link AudioBlockCapabilities} for the current, non-adaptive blocks: they modify audio, work
 * at any sample rate, need no framing/history and support both offline and streaming use. Built with a
 * small fluent builder so later adaptive/analysis blocks can declare framing, profiles or metadata.
 */
public final class StaticBlockCapabilities implements AudioBlockCapabilities {

    private final boolean requiresFraming;
    private final int preferredFrameSize;
    private final int requiredLookAhead;
    private final int requiredHistory;
    private final int[] supportedSampleRates;
    private final boolean modifiesAudio;
    private final boolean producesMetadata;
    private final boolean requiresNoiseProfile;
    private final boolean requiresRoomProfile;
    private final boolean supportsStreaming;
    private final boolean supportsOffline;
    private final boolean preservesChannelCount;
    private final boolean consumesSpeechMetadata;
    private final boolean requiresSpeechActivityTrack;
    private final boolean changesDuration;
    private final boolean changesSampleCount;
    private final boolean requiresCompleteSignal;
    private final boolean requiresSynchronizedChannels;
    private final boolean requiresKnownMicrophoneGeometry;

    private StaticBlockCapabilities(Builder builder) {
        this.requiresFraming = builder.requiresFraming;
        this.preferredFrameSize = builder.preferredFrameSize;
        this.requiredLookAhead = builder.requiredLookAhead;
        this.requiredHistory = builder.requiredHistory;
        this.supportedSampleRates = builder.supportedSampleRates.clone();
        this.modifiesAudio = builder.modifiesAudio;
        this.producesMetadata = builder.producesMetadata;
        this.requiresNoiseProfile = builder.requiresNoiseProfile;
        this.requiresRoomProfile = builder.requiresRoomProfile;
        this.supportsStreaming = builder.supportsStreaming;
        this.supportsOffline = builder.supportsOffline;
        this.preservesChannelCount = builder.preservesChannelCount;
        this.consumesSpeechMetadata = builder.consumesSpeechMetadata;
        this.requiresSpeechActivityTrack = builder.requiresSpeechActivityTrack;
        this.changesDuration = builder.changesDuration;
        this.changesSampleCount = builder.changesSampleCount;
        this.requiresCompleteSignal = builder.requiresCompleteSignal;
        this.requiresSynchronizedChannels = builder.requiresSynchronizedChannels;
        this.requiresKnownMicrophoneGeometry = builder.requiresKnownMicrophoneGeometry;
    }

    /** @return capabilities for a plain audio effect: modifies audio, any rate, no framing, offline+streaming. */
    public static AudioBlockCapabilities audioEffect() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public boolean requiresFraming() {
        return requiresFraming;
    }

    public int getPreferredFrameSize() {
        return preferredFrameSize;
    }

    public int getRequiredLookAhead() {
        return requiredLookAhead;
    }

    public int getRequiredHistory() {
        return requiredHistory;
    }

    public int[] getSupportedSampleRates() {
        return supportedSampleRates.clone();
    }

    public boolean modifiesAudio() {
        return modifiesAudio;
    }

    public boolean producesMetadata() {
        return producesMetadata;
    }

    public boolean requiresNoiseProfile() {
        return requiresNoiseProfile;
    }

    public boolean requiresRoomProfile() {
        return requiresRoomProfile;
    }

    public boolean supportsStreaming() {
        return supportsStreaming;
    }

    public boolean supportsOffline() {
        return supportsOffline;
    }

    public boolean preservesChannelCount() {
        return preservesChannelCount;
    }

    public boolean consumesSpeechMetadata() {
        return consumesSpeechMetadata;
    }

    public boolean requiresSpeechActivityTrack() {
        return requiresSpeechActivityTrack;
    }

    public boolean changesDuration() {
        return changesDuration;
    }

    public boolean changesSampleCount() {
        return changesSampleCount;
    }

    public boolean requiresCompleteSignal() {
        return requiresCompleteSignal;
    }

    public boolean requiresSynchronizedChannels() {
        return requiresSynchronizedChannels;
    }

    public boolean requiresKnownMicrophoneGeometry() {
        return requiresKnownMicrophoneGeometry;
    }

    public static final class Builder {
        private boolean requiresFraming;
        private int preferredFrameSize;
        private int requiredLookAhead;
        private int requiredHistory;
        private int[] supportedSampleRates = new int[0];
        private boolean modifiesAudio = true;
        private boolean producesMetadata;
        private boolean requiresNoiseProfile;
        private boolean requiresRoomProfile;
        private boolean supportsStreaming = true;
        private boolean supportsOffline = true;
        private boolean preservesChannelCount = true;
        private boolean consumesSpeechMetadata;
        private boolean requiresSpeechActivityTrack;
        private boolean changesDuration;
        private boolean changesSampleCount;
        private boolean requiresCompleteSignal;
        private boolean requiresSynchronizedChannels;
        private boolean requiresKnownMicrophoneGeometry;

        public Builder framing(int frameSize, int lookAhead, int history) {
            this.requiresFraming = true;
            this.preferredFrameSize = frameSize;
            this.requiredLookAhead = lookAhead;
            this.requiredHistory = history;
            return this;
        }

        public Builder supportedSampleRates(int... rates) {
            this.supportedSampleRates = rates == null ? new int[0] : rates.clone();
            return this;
        }

        public Builder modifiesAudio(boolean value) {
            this.modifiesAudio = value;
            return this;
        }

        public Builder producesMetadata(boolean value) {
            this.producesMetadata = value;
            return this;
        }

        public Builder requiresNoiseProfile(boolean value) {
            this.requiresNoiseProfile = value;
            return this;
        }

        public Builder requiresRoomProfile(boolean value) {
            this.requiresRoomProfile = value;
            return this;
        }

        public Builder streaming(boolean value) {
            this.supportsStreaming = value;
            return this;
        }

        public Builder offline(boolean value) {
            this.supportsOffline = value;
            return this;
        }

        public Builder preservesChannelCount(boolean value) {
            this.preservesChannelCount = value;
            return this;
        }

        public Builder consumesSpeechMetadata(boolean value) {
            this.consumesSpeechMetadata = value;
            return this;
        }

        public Builder requiresSpeechActivityTrack(boolean value) {
            this.requiresSpeechActivityTrack = value;
            return this;
        }

        public Builder changesDuration(boolean value) {
            this.changesDuration = value;
            return this;
        }

        public Builder changesSampleCount(boolean value) {
            this.changesSampleCount = value;
            return this;
        }

        public Builder requiresCompleteSignal(boolean value) {
            this.requiresCompleteSignal = value;
            return this;
        }

        public Builder requiresSynchronizedChannels(boolean value) {
            this.requiresSynchronizedChannels = value;
            return this;
        }

        public Builder requiresKnownMicrophoneGeometry(boolean value) {
            this.requiresKnownMicrophoneGeometry = value;
            return this;
        }

        public StaticBlockCapabilities build() {
            return new StaticBlockCapabilities(this);
        }
    }
}
