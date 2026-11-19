package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class HighPassFilterProcessorTest {

    @Test
    public void attenuatesRumbleMoreThanSpeechBand() {
        double rumbleRatio = passThroughRatio(30.0d);
        double speechRatio = passThroughRatio(1000.0d);

        assertTrue("30 Hz should be strongly attenuated (ratio was " + rumbleRatio + ")",
                rumbleRatio < 0.6d);
        assertTrue("1000 Hz should pass mostly unchanged (ratio was " + speechRatio + ")",
                speechRatio > 0.9d);
        assertTrue("30 Hz must be attenuated more than 1000 Hz",
                rumbleRatio < speechRatio);
    }

    /** Feed one second of a pure tone through the filter and return RMS(out) / RMS(in). */
    private double passThroughRatio(double frequencyHz) {
        PcmAudioFormat format = PcmAudioFormat.speechDefault();
        HighPassFilterProcessor processor = new HighPassFilterProcessor(80.0d);
        int frameSize = format.samplesForMillis(20);

        double sumInputSquares = 0;
        double sumOutputSquares = 0;
        long counted = 0;
        int sampleIndex = 0;
        for (int frame = 0; frame < 50; frame++) {
            short[] samples = new short[frameSize];
            for (int i = 0; i < frameSize; i++) {
                samples[i] = (short) (8000.0d
                        * Math.sin(2 * Math.PI * frequencyHz * sampleIndex / format.getSampleRateHz()));
                sampleIndex++;
            }
            short[] input = samples.clone();
            processor.process(samples, frameSize, format);
            // Skip the first frames so the filter has settled.
            if (frame >= 5) {
                for (int i = 0; i < frameSize; i++) {
                    sumInputSquares += (double) input[i] * input[i];
                    sumOutputSquares += (double) samples[i] * samples[i];
                }
                counted += frameSize;
            }
        }
        double rmsIn = Math.sqrt(sumInputSquares / counted);
        double rmsOut = Math.sqrt(sumOutputSquares / counted);
        return rmsOut / rmsIn;
    }
}
