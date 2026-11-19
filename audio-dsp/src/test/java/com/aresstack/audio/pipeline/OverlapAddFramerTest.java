package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.PcmAudioFormat;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/** Framed overlap-add must lose no samples and reconstruct an identity processor's input. */
public class OverlapAddFramerTest {

    /** A frame processor that copies input to output unchanged. */
    private static final class IdentityProcessor implements StatefulAudioBlockProcessor {
        public void initialize(PcmAudioFormat format) {
        }

        public void process(short[] input, short[] output) {
            System.arraycopy(input, 0, output, 0, input.length);
        }

        public void reset() {
        }
    }

    @Test
    public void preservesLengthForNonMultipleOfHop() {
        OverlapAddFramer framer = new OverlapAddFramer(256, 128);
        short[] input = ramp(1000); // not a multiple of the hop
        short[] output = framer.process(input, PcmAudioFormat.speechDefault(), new IdentityProcessor());
        assertEquals(input.length, output.length);
    }

    @Test
    public void identityProcessorReconstructsTheInput() {
        OverlapAddFramer framer = new OverlapAddFramer(256, 128);
        short[] input = tone(4096);
        short[] output = framer.process(input, PcmAudioFormat.speechDefault(), new IdentityProcessor());
        assertEquals(input.length, output.length);
        int maxError = 0;
        for (int i = 0; i < input.length; i++) {
            maxError = Math.max(maxError, Math.abs(input[i] - output[i]));
        }
        // Windowed overlap-add with per-sample normalization reconstructs the signal within 16-bit rounding.
        assertTrue("reconstruction error " + maxError + " must stay tiny", maxError <= 2);
    }

    @Test
    public void emptyInputYieldsEmptyOutput() {
        OverlapAddFramer framer = new OverlapAddFramer(64, 32);
        short[] output = framer.process(new short[0], PcmAudioFormat.speechDefault(), new IdentityProcessor());
        assertEquals(0, output.length);
    }

    private static short[] ramp(int length) {
        short[] samples = new short[length];
        for (int i = 0; i < length; i++) {
            samples[i] = (short) ((i % 200) - 100);
        }
        return samples;
    }

    private static short[] tone(int length) {
        short[] samples = new short[length];
        for (int i = 0; i < length; i++) {
            samples[i] = (short) Math.round(8000.0d * Math.sin(2.0d * Math.PI * 220.0d * i / 16000.0d));
        }
        return samples;
    }
}
