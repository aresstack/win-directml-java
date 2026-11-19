package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.dsp.ChannelAligner;
import com.aresstack.audio.dsp.ChannelDiagnostics;
import com.aresstack.audio.dsp.MultichannelOps;
import com.aresstack.audio.profile.AudioBlockType;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/** Slice 10B: channel delay alignment, best-channel selection and channel health analysis. */
public class ChannelAlignmentQualityTest {

    private static final int RATE = 16000;
    private static final PcmAudioFormat STEREO = new PcmAudioFormat(RATE, 2, 16);
    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();

    @Test
    public void estimatesAndAlignsAChannelDelay() {
        short[] base = noise(4000);
        short[] delayed = new short[base.length];
        for (int t = 0; t < base.length; t++) {
            delayed[t] = t >= 5 ? base[t - 5] : 0; // channel 1 lags channel 0 by 5 samples
        }
        short[] stereo = interleave(base, delayed);
        double est = ChannelAligner.estimateDelay(stereo, stereo.length, 2, 0, 1, 64);
        assertTrue("estimated ~5 sample lag: " + est, Math.abs(est - 5.0d) < 1.5d);

        double before = MultichannelOps.correlation(stereo, stereo.length, 2, 0, 1);
        registry.createProcessor(AudioBlockType.CHANNEL_DELAY_ALIGNMENT)
                .process(new AudioBuffer(stereo, STEREO),
                        registry.defaultDefinition(AudioBlockType.CHANNEL_DELAY_ALIGNMENT, "a"),
                        new AudioProcessingContext());
        double after = MultichannelOps.correlation(stereo, stereo.length, 2, 0, 1);
        assertTrue("alignment improves correlation: " + before + " -> " + after, after > before + 0.2d);
    }

    @Test
    public void bestChannelSelectorPicksTheStrongerCleanerChannel() {
        short[] weak = noise2(4000, 100);
        short[] strong = tone(440.0d, 4000, 8000);
        AudioBuffer out = registry.createProcessor(AudioBlockType.BEST_CHANNEL_SELECTOR)
                .process(new AudioBuffer(interleave(weak, strong), STEREO),
                        registry.defaultDefinition(AudioBlockType.BEST_CHANNEL_SELECTOR, "b"),
                        new AudioProcessingContext());
        assertEquals(1, out.getFormat().getChannels());
        assertArrayEquals(strong, out.getSamples());
    }

    @Test
    public void channelHealthAnalyzerFlagsASilentChannel() {
        short[] silent = new short[4000];
        short[] signal = tone(440.0d, 4000, 8000);
        AudioProcessingContext ctx = new AudioProcessingContext();
        registry.createProcessor(AudioBlockType.CHANNEL_HEALTH_ANALYZER)
                .process(new AudioBuffer(interleave(silent, signal), STEREO),
                        registry.defaultDefinition(AudioBlockType.CHANNEL_HEALTH_ANALYZER, "h"), ctx);
        assertNotNull(ctx.getChannelHealthSummary());
        assertTrue("silent channel flagged: " + ctx.getChannelHealthSummary(),
                ctx.getChannelHealthSummary().contains("channel 0 is silent"));
    }

    @Test
    public void healthReportsClippingChannel() {
        short[] clean = tone(300.0d, 4000, 3000);
        short[] clipping = new short[4000];
        for (int i = 0; i < clipping.length; i++) {
            clipping[i] = (short) (i % 2 == 0 ? 32767 : -32767);
        }
        String health = ChannelDiagnostics.describeHealth(interleave(clean, clipping),
                clean.length * 2, 2);
        assertTrue("clipping flagged: " + health, health.contains("channel 1 clips"));
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

    private static short[] noise(int n) {
        return noise2(n, 6000);
    }

    private static short[] noise2(int n, int amp) {
        short[] out = new short[n];
        int state = 555;
        for (int i = 0; i < n; i++) {
            state = state * 1103515245 + 12345;
            out[i] = (short) (((state >> 16) % (2 * amp)) - amp);
        }
        return out;
    }
}
