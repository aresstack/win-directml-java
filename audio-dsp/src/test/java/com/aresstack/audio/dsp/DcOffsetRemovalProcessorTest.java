package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class DcOffsetRemovalProcessorTest {

    @Test
    public void removesConstantOffsetFromSine() {
        PcmAudioFormat format = PcmAudioFormat.speechDefault();
        DcOffsetRemovalProcessor processor = new DcOffsetRemovalProcessor();

        // Generate one second of 200 Hz sine with a constant +2000 offset, in 20 ms frames.
        int frameSize = format.samplesForMillis(20);
        double meanBefore = 0;
        double meanAfter = 0;
        long totalSamples = 0;
        int sampleIndex = 0;
        for (int frame = 0; frame < 50; frame++) {
            short[] samples = new short[frameSize];
            for (int i = 0; i < frameSize; i++) {
                double sine = 4000.0d * Math.sin(2 * Math.PI * 200 * sampleIndex / format.getSampleRateHz());
                samples[i] = (short) (sine + 2000.0d);
                meanBefore += samples[i];
                sampleIndex++;
            }
            processor.process(samples, frameSize, format);
            for (int i = 0; i < frameSize; i++) {
                meanAfter += samples[i];
            }
            totalSamples += frameSize;
        }
        meanBefore /= totalSamples;
        meanAfter /= totalSamples;

        assertTrue("Mean before processing should show the offset (was " + meanBefore + ")",
                Math.abs(meanBefore) > 1500);
        assertTrue("Mean after processing should be clearly reduced (was " + meanAfter + ")",
                Math.abs(meanAfter) < Math.abs(meanBefore) / 4);
    }
}
