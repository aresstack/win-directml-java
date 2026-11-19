package com.aresstack.windirectml.whisper.audio;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.pipeline.AudioBlockRegistry;
import com.aresstack.audio.pipeline.AudioProfileProcessor;
import com.aresstack.audio.profile.AudioBlockDefinition;
import com.aresstack.audio.profile.AudioBlockType;
import com.aresstack.audio.profile.AudioProcessingProfile;
import com.aresstack.windirectml.inference.whisper.WhisperAudioSource;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Adapt Ask AI's 16-bit PCM processing pipeline to one original-timeline speech
 * interval. Keep the recording's original sample offsets unchanged.
 *
 * <p>The processor creates fresh filter state for every call. Each read therefore
 * starts half a second earlier in the original recording and discards that warm-up,
 * so every window equals continuous processing of the whole recording: the 80-Hz
 * high-pass and the DC estimate settle within a few milliseconds.
 */
final class AskAiProcessedSpeechSource implements WhisperAudioSource {
    private static final PcmAudioFormat FORMAT = PcmAudioFormat.speechDefault();
    static final int WARM_UP_SAMPLES = 8000;

    private final WhisperAudioSource original;
    /** The speech interval, or {@code null} for the whole recording while it is still growing. */
    private final SpeechInterval interval;
    private final AudioProfileProcessor processor = new AudioProfileProcessor();
    private final AudioProcessingProfile profile;

    AskAiProcessedSpeechSource(WhisperAudioSource original, SpeechInterval interval,
                               AudioProcessingProfile profile) {
        this.original = Objects.requireNonNull(original, "original");
        this.interval = interval;
        this.profile = Objects.requireNonNull(profile, "profile");
    }

    static AudioProcessingProfile vadProfile() {
        AudioBlockRegistry registry = AudioBlockRegistry.getInstance();
        List<AudioBlockDefinition> blocks = new ArrayList<>();
        blocks.add(registry.defaultDefinition(AudioBlockType.DC_OFFSET_REMOVAL, "vad-dc-offset"));
        blocks.add(highPass(registry, "vad-high-pass"));
        return new AudioProcessingProfile("whisper-vad", "VAD cleanup", true, blocks);
    }

    /**
     * Remove DC offset and rumble below 80 Hz only. Both leave the speech band untouched;
     * gates, compressors, limiters and low-pass filters change what Whisper hears.
     */
    static AudioProcessingProfile speechProfile(boolean highPass) {
        AudioBlockRegistry registry = AudioBlockRegistry.getInstance();
        List<AudioBlockDefinition> blocks = new ArrayList<>();
        blocks.add(registry.defaultDefinition(AudioBlockType.DC_OFFSET_REMOVAL, "whisper-dc-offset"));
        if (highPass) {
            blocks.add(highPass(registry, "whisper-high-pass"));
        }
        return new AudioProcessingProfile("whisper-speech", "Whisper speech preparation", true, blocks);
    }

    /** Reproduce the 0.3.1 cleanup chain, including gate, compressor, limiter and low-pass. */
    static AudioProcessingProfile enhancedSpeechProfile() {
        AudioBlockRegistry registry = AudioBlockRegistry.getInstance();
        List<AudioBlockDefinition> blocks = new ArrayList<>();
        blocks.add(registry.defaultDefinition(AudioBlockType.DC_OFFSET_REMOVAL, "whisper-dc-offset"));
        blocks.add(highPass(registry, "whisper-high-pass"));
        blocks.add(registry.defaultDefinition(AudioBlockType.LOW_PASS, "whisper-low-pass")
                .withParameter("cutoffHz", "7200"));
        blocks.add(registry.defaultDefinition(AudioBlockType.NOISE_GATE, "whisper-noise-gate")
                .withParameter("threshold", "180")
                .withParameter("closedGain", "0.4"));
        blocks.add(registry.defaultDefinition(AudioBlockType.COMPRESSOR, "whisper-compressor"));
        blocks.add(registry.defaultDefinition(AudioBlockType.LIMITER, "whisper-limiter"));
        return new AudioProcessingProfile("whisper-enhanced", "Whisper speech cleanup", true, blocks);
    }

    private static AudioBlockDefinition highPass(AudioBlockRegistry registry, String blockId) {
        return registry.defaultDefinition(AudioBlockType.HIGH_PASS, blockId)
                .withParameter("cutoffHz", "80")
                .withParameter("implementation", "BUTTERWORTH")
                .withParameter("order", "2");
    }

    @Override
    public long sampleCount() {
        return interval == null ? original.sampleCount() : interval.endSample() - interval.startSample();
    }

    @Override
    public float[] readSamples(long startSample, int sampleCount) throws IOException {
        if (startSample < 0 || sampleCount < 0 || startSample > sampleCount() - sampleCount) {
            throw new IllegalArgumentException("Speech interval request exceeds its sample range");
        }
        long absoluteStart = (interval == null ? 0 : interval.startSample()) + startSample;
        int warmUp = (int) Math.min(WARM_UP_SAMPLES, absoluteStart);
        int total = warmUp + sampleCount;
        float[] samples = original.readSamples(absoluteStart - warmUp, total);
        if (samples.length != total) {
            throw new IOException("Audio source returned an incomplete speech interval");
        }
        short[] pcm = new short[total];
        for (int i = 0; i < total; i++) {
            pcm[i] = SpeechRegionDetector.toPcm16(samples[i]);
        }
        AudioBuffer result = processor.process(new AudioBuffer(pcm, FORMAT), profile);
        short[] processed = result.getSamples();
        if (result.getFormat().getSampleRateHz() != FORMAT.getSampleRateHz()
                || result.getFormat().getChannels() != 1
                || processed.length != total) {
            throw new IOException("Ask AI speech profile must preserve the sample timebase");
        }
        float[] normalized = new float[sampleCount];
        for (int i = 0; i < sampleCount; i++) {
            normalized[i] = processed[warmUp + i] / 32768.0f;
        }
        return normalized;
    }
}
