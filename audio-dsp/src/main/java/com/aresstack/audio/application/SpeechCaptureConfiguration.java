package com.aresstack.audio.application;

import java.io.File;

/**
 * Bundle every knob of the speech capture chain. Immutable; build via {@link Builder}. The
 * defaults are tuned for speech-to-text: 16 kHz mono 16-bit, 20 ms frames, gentle processing
 * that cleans the signal without colouring the voice.
 */
public final class SpeechCaptureConfiguration {

    private final int sampleRateHz;
    private final int channels;
    private final int bitsPerSample;
    private final int frameDurationMillis;
    private final double highPassCutoffHz;
    private final double noiseGateThreshold;
    private final double noiseGateClosedGain;
    private final double compressorThreshold;
    private final double compressorRatio;
    private final int limiterCeiling;
    private final File outputFile;
    private final String deviceName;

    private SpeechCaptureConfiguration(Builder builder) {
        this.sampleRateHz = builder.sampleRateHz;
        this.channels = builder.channels;
        this.bitsPerSample = builder.bitsPerSample;
        this.frameDurationMillis = builder.frameDurationMillis;
        this.highPassCutoffHz = builder.highPassCutoffHz;
        this.noiseGateThreshold = builder.noiseGateThreshold;
        this.noiseGateClosedGain = builder.noiseGateClosedGain;
        this.compressorThreshold = builder.compressorThreshold;
        this.compressorRatio = builder.compressorRatio;
        this.limiterCeiling = builder.limiterCeiling;
        this.outputFile = builder.outputFile;
        this.deviceName = builder.deviceName;
    }

    /** Create the default speech configuration described in the class comment. */
    public static SpeechCaptureConfiguration speechDefaults() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public int getSampleRateHz() {
        return sampleRateHz;
    }

    public int getChannels() {
        return channels;
    }

    public int getBitsPerSample() {
        return bitsPerSample;
    }

    public int getFrameDurationMillis() {
        return frameDurationMillis;
    }

    public double getHighPassCutoffHz() {
        return highPassCutoffHz;
    }

    public double getNoiseGateThreshold() {
        return noiseGateThreshold;
    }

    public double getNoiseGateClosedGain() {
        return noiseGateClosedGain;
    }

    public double getCompressorThreshold() {
        return compressorThreshold;
    }

    public double getCompressorRatio() {
        return compressorRatio;
    }

    public int getLimiterCeiling() {
        return limiterCeiling;
    }

    public File getOutputFile() {
        return outputFile;
    }

    /** @return the capture device name, or {@code null} for the system default microphone. */
    public String getDeviceName() {
        return deviceName;
    }

    public static final class Builder {

        private int sampleRateHz = 16000;
        private int channels = 1;
        private int bitsPerSample = 16;
        private int frameDurationMillis = 20;
        private double highPassCutoffHz = 80.0d;
        private double noiseGateThreshold = 300.0d;
        private double noiseGateClosedGain = 0.3d;
        private double compressorThreshold = 12000.0d;
        private double compressorRatio = 3.0d;
        private int limiterCeiling = 30000;
        private File outputFile = new File("speech-input.wav");
        private String deviceName;

        public Builder sampleRateHz(int value) {
            this.sampleRateHz = value;
            return this;
        }

        public Builder channels(int value) {
            this.channels = value;
            return this;
        }

        public Builder bitsPerSample(int value) {
            this.bitsPerSample = value;
            return this;
        }

        public Builder frameDurationMillis(int value) {
            this.frameDurationMillis = value;
            return this;
        }

        public Builder highPassCutoffHz(double value) {
            this.highPassCutoffHz = value;
            return this;
        }

        public Builder noiseGateThreshold(double value) {
            this.noiseGateThreshold = value;
            return this;
        }

        public Builder noiseGateClosedGain(double value) {
            this.noiseGateClosedGain = value;
            return this;
        }

        public Builder compressorThreshold(double value) {
            this.compressorThreshold = value;
            return this;
        }

        public Builder compressorRatio(double value) {
            this.compressorRatio = value;
            return this;
        }

        public Builder limiterCeiling(int value) {
            this.limiterCeiling = value;
            return this;
        }

        public Builder outputFile(File value) {
            this.outputFile = value;
            return this;
        }

        public Builder deviceName(String value) {
            this.deviceName = value;
            return this;
        }

        public SpeechCaptureConfiguration build() {
            return new SpeechCaptureConfiguration(this);
        }
    }
}
