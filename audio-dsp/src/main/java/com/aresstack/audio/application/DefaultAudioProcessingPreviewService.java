package com.aresstack.audio.application;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.pipeline.AudioProfileProcessor;
import com.aresstack.audio.profile.AudioProcessingProfile;

/**
 * Preview via the productive {@link AudioProfileProcessor}. A fresh processor is created per call, so any
 * adaptive block starts from a clean state for every independent test run; the source buffer is not
 * modified (the processor copies its input).
 */
public final class DefaultAudioProcessingPreviewService implements AudioProcessingPreviewService {

    public ProcessedAudioPreview process(AudioBuffer source, AudioProcessingProfile profile, String sourceId) {
        if (source == null) {
            throw new IllegalArgumentException("Source must not be null.");
        }
        if (profile == null) {
            throw new IllegalArgumentException("Profile must not be null.");
        }
        AudioBuffer processed = new AudioProfileProcessor().process(source, profile);
        PcmAudioFormat format = processed.getFormat();
        int channels = format.getChannels();
        long frames = channels > 0 ? (long) processed.getSamples().length / channels : 0L;
        long durationMillis = format.getSampleRateHz() > 0
                ? frames * 1000L / format.getSampleRateHz() : 0L;
        return new ProcessedAudioPreview(processed.getSamples(), format, durationMillis, sourceId,
                AudioProfileSignature.of(profile));
    }
}
