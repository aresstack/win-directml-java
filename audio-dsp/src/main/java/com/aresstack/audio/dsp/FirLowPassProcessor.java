package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/** Apply the existing 65-tap linear-phase FIR low-pass filter as an in-place pipeline processor. */
public final class FirLowPassProcessor implements Pcm16Processor {

    private final double cutoffHz;

    public FirLowPassProcessor(double cutoffHz) {
        if (cutoffHz <= 0.0d) {
            throw new IllegalArgumentException("Cutoff frequency must be positive.");
        }
        this.cutoffHz = cutoffHz;
    }

    @Override
    public void process(short[] samples, int sampleCount, PcmAudioFormat format) {
        int channels = format.getChannels();
        if (channels == 1) {
            short[] input = copyValidSamples(samples, sampleCount);
            short[] filtered = Pcm16LowPassFilter.filter(input, format.getSampleRateHz(), cutoffHz);
            System.arraycopy(filtered, 0, samples, 0, filtered.length);
            return;
        }
        processInterleavedChannels(samples, sampleCount, format);
    }

    private void processInterleavedChannels(short[] samples, int sampleCount, PcmAudioFormat format) {
        int channels = format.getChannels();
        int frameCount = sampleCount / channels;
        for (int channel = 0; channel < channels; channel++) {
            short[] channelSamples = new short[frameCount];
            for (int frame = 0; frame < frameCount; frame++) {
                channelSamples[frame] = samples[frame * channels + channel];
            }
            short[] filtered = Pcm16LowPassFilter.filter(
                    channelSamples, format.getSampleRateHz(), cutoffHz);
            for (int frame = 0; frame < frameCount; frame++) {
                samples[frame * channels + channel] = filtered[frame];
            }
        }
    }

    private static short[] copyValidSamples(short[] samples, int sampleCount) {
        short[] copy = new short[sampleCount];
        System.arraycopy(samples, 0, copy, 0, sampleCount);
        return copy;
    }
}
