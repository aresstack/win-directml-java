package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Protect against clipping at the very end of the pipeline: clamp every sample whose magnitude
 * exceeds the ceiling to exactly the ceiling. Samples below the ceiling pass unchanged.
 */
public final class LimiterProcessor implements Pcm16Processor {

    private final short ceiling;

    /** @param ceiling maximum allowed magnitude, e.g. 30000 to leave headroom below 32767 */
    public LimiterProcessor(int ceiling) {
        if (ceiling <= 0 || ceiling > Short.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "Ceiling must be within (0, " + Short.MAX_VALUE + "]: " + ceiling);
        }
        this.ceiling = (short) ceiling;
    }

    @Override
    public void process(short[] samples, int sampleCount, PcmAudioFormat format) {
        for (int i = 0; i < sampleCount; i++) {
            if (samples[i] > ceiling) {
                samples[i] = ceiling;
            } else if (samples[i] < -ceiling) {
                samples[i] = (short) -ceiling;
            }
        }
    }
}
