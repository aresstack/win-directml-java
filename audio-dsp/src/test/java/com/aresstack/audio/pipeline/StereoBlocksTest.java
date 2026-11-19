package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.profile.AudioBlockDefinition;
import com.aresstack.audio.profile.AudioBlockType;

import org.junit.Test;

import static org.junit.Assert.assertTrue;

/** Slice 10C: Mid/Side Processor, Center Speech Extractor and Stereo Width Control. */
public class StereoBlocksTest {

    private static final int RATE = 16000;
    private static final PcmAudioFormat STEREO = new PcmAudioFormat(RATE, 2, 16);
    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();

    @Test
    public void midSideSideReductionRemovesSideKeepsCentre() {
        short[] tone = tone(440.0d, 4000, 8000);
        short[] side = run(AudioBlockType.MID_SIDE_PROCESSOR,
                registry.defaultDefinition(AudioBlockType.MID_SIDE_PROCESSOR, "m")
                        .withParameter("sideReduction", "1"),
                interleave(tone, negate(tone))); // pure side: L = -R
        assertTrue("pure side removed", energy(side) < 0.02d * energy(interleave(tone, negate(tone))));

        short[] centre = run(AudioBlockType.MID_SIDE_PROCESSOR,
                registry.defaultDefinition(AudioBlockType.MID_SIDE_PROCESSOR, "m")
                        .withParameter("sideReduction", "1"),
                interleave(tone, tone)); // pure mid: L = R
        assertTrue("pure centre preserved", energy(centre) > 0.9d * energy(interleave(tone, tone)));
    }

    @Test
    public void centerExtractorRemovesLateralContent() {
        short[] tone = tone(440.0d, 4000, 8000);
        short[] out = run(AudioBlockType.CENTER_SPEECH_EXTRACTOR,
                registry.defaultDefinition(AudioBlockType.CENTER_SPEECH_EXTRACTOR, "c")
                        .withParameter("centerAmount", "1"),
                interleave(tone, negate(tone)));
        assertTrue("lateral content removed", energy(out) < 0.02d * energy(interleave(tone, negate(tone))));
    }

    @Test
    public void stereoWidthZeroCollapsesToMono() {
        short[] left = tone(300.0d, 4000, 8000);
        short[] right = tone(1000.0d, 4000, 6000);
        AudioBuffer out = registry.createProcessor(AudioBlockType.STEREO_WIDTH_CONTROL)
                .process(new AudioBuffer(interleave(left, right), STEREO),
                        registry.defaultDefinition(AudioBlockType.STEREO_WIDTH_CONTROL, "w")
                                .withParameter("width", "0"), new AudioProcessingContext());
        short[] s = out.getSamples();
        for (int f = 0; f < s.length / 2; f++) {
            assertTrue("width 0 -> L==R at " + f, Math.abs(s[2 * f] - s[2 * f + 1]) <= 1);
        }
    }

    private short[] run(AudioBlockType type, AudioBlockDefinition block, short[] interleaved) {
        return registry.createProcessor(type)
                .process(new AudioBuffer(interleaved, STEREO), block, new AudioProcessingContext())
                .getSamples();
    }

    private static short[] negate(short[] a) {
        short[] out = new short[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = (short) -a[i];
        }
        return out;
    }

    private static double energy(short[] s) {
        double sum = 0.0d;
        for (short v : s) {
            sum += (double) v * v;
        }
        return sum;
    }

    private static short[] interleave(short[] left, short[] right) {
        short[] out = new short[left.length * 2];
        for (int i = 0; i < left.length; i++) {
            out[2 * i] = left[i];
            out[2 * i + 1] = right[i];
        }
        return out;
    }

    private static short[] tone(double freq, int n, int amp) {
        short[] out = new short[n];
        for (int i = 0; i < n; i++) {
            out[i] = (short) Math.round(amp * Math.sin(2.0d * Math.PI * freq * i / RATE));
        }
        return out;
    }
}
