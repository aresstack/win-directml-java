package com.aresstack.audio.application;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.dsp.AudioLevelMeter;
import com.aresstack.audio.pipeline.AudioProfileProcessor;
import com.aresstack.audio.infrastructure.WavFileReader;
import com.aresstack.audio.profile.AudioProcessingProfile;
import com.aresstack.audio.pipeline.AudioProcessingProfiles;

import java.io.File;
import java.io.IOException;

/**
 * Run a raw recording WAV through the selected reusable audio-processing profile (the DSP stage, which is
 * format-neutral) and then hand the result to a {@link SpeechToTextAudioPreparer} that writes the STT
 * transport WAV. The recording reaches the model as unaltered as the pipeline left it: rate and channels
 * change <b>only</b> through explicit DSP blocks (e.g. a resampler/channel mixer in a cleaning profile),
 * never through a forced final conversion. A pass-through "Off" profile therefore keeps the source format.
 */
public final class SpeechAudioNormalizer {

    private final AudioProcessingProfile profile;
    private final AudioProfileProcessor processor;
    private final SpeechToTextAudioPreparer preparer;

    /** Use the immutable built-in speech profile. */
    public SpeechAudioNormalizer() {
        this(AudioProcessingProfiles.defaultSpeech());
    }

    public SpeechAudioNormalizer(AudioProcessingProfile profile) {
        this(profile, new WavSpeechAudioPreparer());
    }

    public SpeechAudioNormalizer(AudioProcessingProfile profile, SpeechToTextAudioPreparer preparer) {
        if (profile == null) {
            throw new IllegalArgumentException("Profile must not be null.");
        }
        if (preparer == null) {
            throw new IllegalArgumentException("Preparer must not be null.");
        }
        this.profile = profile;
        this.processor = new AudioProfileProcessor();
        this.preparer = preparer;
    }

    public AudioProcessingProfile getProfile() {
        return profile;
    }

    /**
     * @param rawWav    the recorded WAV in the negotiated capture format
     * @param targetWav where to write the profile output
     * @return the written file plus source/target formats, duration and raw-signal level stats
     */
    public NormalizationResult normalize(File rawWav, File targetWav) throws IOException {
        WavFileReader.WavData raw = WavFileReader.read(rawWav);
        PcmAudioFormat sourceFormat = raw.getFormat();
        short[] rawSamples = raw.getSamples();

        AudioLevelMeter rawMeter = new AudioLevelMeter();
        rawMeter.process(rawSamples, rawSamples.length, sourceFormat);

        int channels = sourceFormat.getChannels();
        long frames = channels > 0 ? (long) rawSamples.length / channels : 0L;
        long durationMillis = sourceFormat.getSampleRateHz() > 0
                ? frames * 1000L / sourceFormat.getSampleRateHz() : 0L;

        // DSP stage: apply the selected profile, format-neutral (rate/channels change only via blocks).
        AudioBuffer processed = processor.process(new AudioBuffer(rawSamples, sourceFormat), profile);
        // Final STT transport stage: write the result as WAV, preserving its (source or block-changed) format.
        File written = preparer.prepare(processed, targetWav);

        return new NormalizationResult(written, sourceFormat, processed.getFormat(), durationMillis,
                rawMeter.getOverallRms(), rawMeter.getPeak(), rawMeter.getClippedSampleCount(),
                rawMeter.getTotalSampleCount());
    }
}
