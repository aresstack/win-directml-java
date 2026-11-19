package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.profile.AudioBlockDefinition;
import com.aresstack.audio.profile.AudioBlockType;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/** The Slice 10A multichannel blocks select, downmix, gain/polarity and analyze channels correctly. */
public class MultichannelBlocksTest {

    private static final int RATE = 16000;
    private static final PcmAudioFormat STEREO = new PcmAudioFormat(RATE, 2, 16);
    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();

    @Test
    public void channelSelectorPicksTheRequestedChannelAsMono() {
        short[] left = tone(300.0d, 4000, 8000);
        short[] right = tone(1000.0d, 4000, 4000);
        AudioBuffer out = run(AudioBlockType.CHANNEL_SELECTOR,
                registry.defaultDefinition(AudioBlockType.CHANNEL_SELECTOR, "s")
                        .withParameter("channelIndex", "1"),
                interleave(left, right));
        assertEquals(1, out.getFormat().getChannels());
        assertArrayEquals(right, out.getSamples());
    }

    @Test
    public void matrixMixerDownmixesWithWeights() {
        short[] left = tone(300.0d, 4000, 8000);
        short[] right = tone(1000.0d, 4000, 4000);
        AudioBuffer leftOnly = run(AudioBlockType.MATRIX_MIXER,
                registry.defaultDefinition(AudioBlockType.MATRIX_MIXER, "m")
                        .withParameter("weights", "1,0").withParameter("normalize", "false"),
                interleave(left, right));
        assertEquals(1, leftOnly.getFormat().getChannels());
        assertArrayEquals(left, leftOnly.getSamples());
    }

    @Test
    public void oppositePolarityWeightsCancelAnIdenticalStereoSignal() {
        short[] mono = tone(440.0d, 4000, 8000);
        AudioBuffer out = run(AudioBlockType.MATRIX_MIXER,
                registry.defaultDefinition(AudioBlockType.MATRIX_MIXER, "m")
                        .withParameter("weights", "0.5,-0.5").withParameter("normalize", "false"),
                interleave(mono, mono));
        for (short s : out.getSamples()) {
            assertTrue("cancels to near zero: " + s, Math.abs(s) <= 1);
        }
    }

    @Test
    public void correlationAnalyzerDetectsIdenticalAndInvertedChannels() {
        short[] mono = tone(440.0d, 4000, 8000);
        AudioProcessingContext same = new AudioProcessingContext();
        registry.createProcessor(AudioBlockType.PHASE_CORRELATION_ANALYZER)
                .process(new AudioBuffer(interleave(mono, mono), STEREO),
                        registry.defaultDefinition(AudioBlockType.PHASE_CORRELATION_ANALYZER, "p"), same);
        assertNotNull(same.getChannelCorrelation());
        assertTrue("identical channels correlate ~1: " + same.getChannelCorrelation(),
                same.getChannelCorrelation() > 0.95d);

        short[] inverted = new short[mono.length];
        for (int i = 0; i < mono.length; i++) {
            inverted[i] = (short) -mono[i];
        }
        AudioProcessingContext opp = new AudioProcessingContext();
        registry.createProcessor(AudioBlockType.PHASE_CORRELATION_ANALYZER)
                .process(new AudioBuffer(interleave(mono, inverted), STEREO),
                        registry.defaultDefinition(AudioBlockType.PHASE_CORRELATION_ANALYZER, "p"), opp);
        assertTrue("inverted channels correlate ~-1: " + opp.getChannelCorrelation(),
                opp.getChannelCorrelation() < -0.95d);
    }

    @Test
    public void channelGainPolarityInvertsAChannelAndPreservesChannelCount() {
        short[] mono = tone(440.0d, 4000, 8000);
        AudioBuffer out = run(AudioBlockType.CHANNEL_GAIN_POLARITY,
                registry.defaultDefinition(AudioBlockType.CHANNEL_GAIN_POLARITY, "g")
                        .withParameter("gainsDb", "0,0").withParameter("polarityInvert", "0,1"),
                interleave(mono, mono));
        assertEquals(2, out.getFormat().getChannels());
        double corr = com.aresstack.audio.dsp.MultichannelOps.correlation(out.getSamples(),
                out.getSamples().length, 2, 0, 1);
        assertTrue("channel 1 inverted -> correlation ~-1: " + corr, corr < -0.95d);
    }

    private AudioBuffer run(AudioBlockType type, AudioBlockDefinition block, short[] interleaved) {
        return registry.createProcessor(type)
                .process(new AudioBuffer(interleaved, STEREO), block, new AudioProcessingContext());
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
