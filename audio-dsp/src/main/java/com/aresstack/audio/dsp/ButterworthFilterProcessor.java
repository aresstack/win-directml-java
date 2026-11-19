package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;
import uk.me.berndporr.iirj.Butterworth;

/** Apply configurable Butterworth filters through the focused iirj DSP library. */
public final class ButterworthFilterProcessor implements Pcm16Processor {

    public enum Kind {
        LOW_PASS,
        HIGH_PASS,
        BAND_PASS,
        BAND_STOP
    }

    private final Kind kind;
    private final int order;
    private final double frequencyHz;
    private final double widthHz;

    private Butterworth[] filters;
    private int configuredSampleRate;
    private int configuredChannels;
    private boolean bypassedForCurrentFormat;

    private ButterworthFilterProcessor(Kind kind, int order, double frequencyHz, double widthHz) {
        if (kind == null) {
            throw new IllegalArgumentException("Filter kind must not be null.");
        }
        if (order < 1 || order > 12) {
            throw new IllegalArgumentException("Filter order must be within [1, 12].");
        }
        if (frequencyHz <= 0.0d) {
            throw new IllegalArgumentException("Filter frequency must be positive.");
        }
        this.kind = kind;
        this.order = order;
        this.frequencyHz = frequencyHz;
        this.widthHz = widthHz;
    }

    public static ButterworthFilterProcessor lowPass(int order, double cutoffHz) {
        return new ButterworthFilterProcessor(Kind.LOW_PASS, order, cutoffHz, 0.0d);
    }

    public static ButterworthFilterProcessor highPass(int order, double cutoffHz) {
        return new ButterworthFilterProcessor(Kind.HIGH_PASS, order, cutoffHz, 0.0d);
    }

    public static ButterworthFilterProcessor bandPass(int order, double centerHz, double widthHz) {
        return new ButterworthFilterProcessor(Kind.BAND_PASS, order, centerHz, widthHz);
    }

    public static ButterworthFilterProcessor bandStop(int order, double centerHz, double widthHz) {
        return new ButterworthFilterProcessor(Kind.BAND_STOP, order, centerHz, widthHz);
    }

    @Override
    public void process(short[] samples, int sampleCount, PcmAudioFormat format) {
        configureFor(format);
        if (bypassedForCurrentFormat) {
            return;
        }
        int channels = format.getChannels();
        for (int i = 0; i < sampleCount; i++) {
            int channel = channels == 1 ? 0 : i % channels;
            samples[i] = clamp(filters[channel].filter(samples[i]));
        }
    }

    private void configureFor(PcmAudioFormat format) {
        if (filters != null && configuredSampleRate == format.getSampleRateHz()
                && configuredChannels == format.getChannels()) {
            return;
        }
        configuredSampleRate = format.getSampleRateHz();
        configuredChannels = format.getChannels();
        bypassedForCurrentFormat = shouldBypass(configuredSampleRate);
        if (bypassedForCurrentFormat) {
            filters = new Butterworth[0];
            return;
        }
        validateFrequency(configuredSampleRate);
        filters = new Butterworth[configuredChannels];
        for (int i = 0; i < filters.length; i++) {
            Butterworth filter = new Butterworth();
            if (kind == Kind.LOW_PASS) {
                filter.lowPass(order, configuredSampleRate, frequencyHz);
            } else if (kind == Kind.HIGH_PASS) {
                filter.highPass(order, configuredSampleRate, frequencyHz);
            } else if (kind == Kind.BAND_PASS) {
                filter.bandPass(order, configuredSampleRate, frequencyHz, widthHz);
            } else {
                filter.bandStop(order, configuredSampleRate, frequencyHz, widthHz);
            }
            filters[i] = filter;
        }
    }

    private boolean shouldBypass(int sampleRateHz) {
        return kind == Kind.LOW_PASS && frequencyHz >= sampleRateHz / 2.0d;
    }

    private void validateFrequency(int sampleRateHz) {
        double nyquist = sampleRateHz / 2.0d;
        if (frequencyHz >= nyquist) {
            throw new IllegalArgumentException("Filter frequency " + frequencyHz
                    + " Hz must stay below Nyquist " + nyquist + " Hz.");
        }
        if ((kind == Kind.BAND_PASS || kind == Kind.BAND_STOP)
                && (widthHz <= 0.0d || frequencyHz - widthHz / 2.0d <= 0.0d
                || frequencyHz + widthHz / 2.0d >= nyquist)) {
            throw new IllegalArgumentException("Band filter must remain within (0, Nyquist).");
        }
    }

    private static short clamp(double value) {
        if (value > Short.MAX_VALUE) {
            return Short.MAX_VALUE;
        }
        if (value < Short.MIN_VALUE) {
            return Short.MIN_VALUE;
        }
        return (short) Math.round(value);
    }
}
