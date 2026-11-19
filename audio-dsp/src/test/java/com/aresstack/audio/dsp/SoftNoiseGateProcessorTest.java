package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertTrue;

public class SoftNoiseGateProcessorTest {

    private static final double THRESHOLD_RMS = 300.0d;
    private static final double CLOSED_GAIN = 0.3d;

    @Test
    public void attenuatesQuietNoise() {
        double ratio = processTone(100.0d);
        assertTrue("Quiet noise should be attenuated toward the closed gain (ratio was " + ratio + ")",
                ratio < 0.7d);
    }

    @Test
    public void keepsLoudSpeech() {
        double ratio = processTone(8000.0d);
        assertTrue("Loud speech must stay essentially unchanged (ratio was " + ratio + ")",
                ratio > 0.9d);
    }

    /** Run one second of noise at the given amplitude through the gate; return RMS(out)/RMS(in). */
    private double processTone(double amplitude) {
        PcmAudioFormat format = PcmAudioFormat.speechDefault();
        SoftNoiseGateProcessor processor =
                new SoftNoiseGateProcessor(THRESHOLD_RMS, CLOSED_GAIN, 5.0d, 150.0d);
        int frameSize = format.samplesForMillis(20);
        Random random = new Random(42);

        double sumInputSquares = 0;
        double sumOutputSquares = 0;
        long counted = 0;
        for (int frame = 0; frame < 50; frame++) {
            short[] samples = new short[frameSize];
            for (int i = 0; i < frameSize; i++) {
                samples[i] = (short) ((random.nextDouble() * 2 - 1) * amplitude);
            }
            short[] input = samples.clone();
            processor.process(samples, frameSize, format);
            // Skip the first frames so the smoothed gain has settled.
            if (frame >= 15) {
                for (int i = 0; i < frameSize; i++) {
                    sumInputSquares += (double) input[i] * input[i];
                    sumOutputSquares += (double) samples[i] * samples[i];
                }
                counted += frameSize;
            }
        }
        return Math.sqrt(sumOutputSquares / counted) / Math.sqrt(sumInputSquares / counted);
    }
}
