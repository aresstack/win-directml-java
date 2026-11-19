package com.aresstack.audio.application;

import com.aresstack.audio.dsp.AudioLevelMeter;
import com.aresstack.audio.dsp.AudioProcessingPipeline;
import com.aresstack.audio.dsp.CompressorProcessor;
import com.aresstack.audio.dsp.DcOffsetRemovalProcessor;
import com.aresstack.audio.dsp.HighPassFilterProcessor;
import com.aresstack.audio.dsp.LimiterProcessor;
import com.aresstack.audio.dsp.Pcm16Processor;
import com.aresstack.audio.dsp.Pcm16ProcessingPipeline;
import com.aresstack.audio.dsp.SoftNoiseGateProcessor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Record speech from an audio source, clean it with the standard DSP chain and hand the frames
 * to a sink. Source and sink are injected, so the same use case records to a WAV file today and
 * streams to a remote speech-to-text endpoint tomorrow.
 */
public final class RecordSpeechInputUseCase {

    private final AudioSource source;
    private final AudioSink sink;
    private final SpeechCaptureConfiguration configuration;
    private final AudioLevelMeter levelMeter = new AudioLevelMeter();

    public RecordSpeechInputUseCase(AudioSource source, AudioSink sink,
                                    SpeechCaptureConfiguration configuration) {
        this.source = source;
        this.sink = sink;
        this.configuration = configuration == null
                ? SpeechCaptureConfiguration.speechDefaults() : configuration;
    }

    /**
     * Record for the given duration and block until the sink is closed.
     *
     * @return the level meter with RMS/peak/clipping measurements of the processed signal
     */
    public AudioLevelMeter record(long durationMillis) throws AudioCaptureException, IOException {
        levelMeter.reset();
        AudioProcessingPipeline pipeline = buildSpeechPipeline(configuration, levelMeter);
        long dropped = new AudioCaptureService().capture(
                source, pipeline, sink, configuration.getFrameDurationMillis(), durationMillis);
        if (dropped > 0) {
            throw new IOException("The sink was too slow: " + dropped
                    + " audio frame(s) were dropped during recording.");
        }
        return levelMeter;
    }

    /**
     * Assemble the standard speech chain: DC offset removal, high-pass, soft noise gate,
     * compressor, limiter, and finally the level meter observing the processed signal.
     */
    public static AudioProcessingPipeline buildSpeechPipeline(SpeechCaptureConfiguration configuration,
                                                              AudioLevelMeter levelMeter) {
        List<Pcm16Processor> processors = new ArrayList<Pcm16Processor>();
        processors.add(new DcOffsetRemovalProcessor());
        processors.add(new HighPassFilterProcessor(configuration.getHighPassCutoffHz()));
        processors.add(new SoftNoiseGateProcessor(
                configuration.getNoiseGateThreshold(),
                configuration.getNoiseGateClosedGain(),
                5.0d, 150.0d));
        processors.add(new CompressorProcessor(
                configuration.getCompressorThreshold(),
                configuration.getCompressorRatio(),
                5.0d, 100.0d));
        processors.add(new LimiterProcessor(configuration.getLimiterCeiling()));
        if (levelMeter != null) {
            processors.add(levelMeter);
        }
        return new Pcm16ProcessingPipeline(processors);
    }
}
