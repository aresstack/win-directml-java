package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.pipeline.AudioBlockRegistry;
import com.aresstack.audio.pipeline.AudioProcessingContext;
import com.aresstack.audio.profile.AudioBlockType;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/** The Room/Reverb Analyzer estimates a larger reverb time for a reverberant signal than a dry one. */
public class RoomReverbAnalyzerTest {

    private static final int RATE = 16000;
    private static final PcmAudioFormat MONO = new PcmAudioFormat(RATE, 1, 16);

    @Test
    public void estimatesMoreReverbForALongerDecay() {
        RoomReverbAnalyzer analyzer = new RoomReverbAnalyzer(20.0d, 6.0d, 3.0d);
        RoomProfile dry = analyzer.analyze(bursts(0.002d), 48000, MONO);
        RoomProfile wet = analyzer.analyze(bursts(0.15d), 48000, MONO);
        assertTrue("wet RT60 > dry RT60: " + wet.getReverbTimeSeconds() + " vs " + dry.getReverbTimeSeconds(),
                wet.getReverbTimeSeconds() > dry.getReverbTimeSeconds());
        assertTrue("wet strength > dry strength", wet.getReverbStrength() > dry.getReverbStrength());
        assertTrue("wet is judged reverberant", wet.getReverbStrength() > 0.3d);
    }

    @Test
    public void publishesProfileAndLeavesAudioUnchanged() {
        AudioProcessingContext ctx = new AudioProcessingContext();
        short[] input = bursts(0.12d);
        short[] out = AudioBlockRegistry.getInstance().createProcessor(AudioBlockType.ROOM_REVERB_ANALYZER)
                .process(new AudioBuffer(input.clone(), MONO),
                        AudioBlockRegistry.getInstance().defaultDefinition(AudioBlockType.ROOM_REVERB_ANALYZER, "r"),
                        ctx).getSamples();
        assertArrayEquals("analyzer must not change audio", input, out);
        assertNotNull(ctx.getRoomProfile());
        assertTrue(ctx.getRoomProfile().getReverbTimeSeconds() >= 0.0d);
    }

    /** Four noise bursts that decay with the given time constant, separated by silence. */
    private static short[] bursts(double tauSeconds) {
        short[] out = new short[48000];
        double tau = tauSeconds * RATE;
        int state = 9173;
        for (int b = 0; b < 4; b++) {
            int start = b * 12000;
            for (int i = 0; i < 9000; i++) {
                state = state * 1103515245 + 12345;
                double n = ((state >> 16) % 2000) - 1000.0d;
                double env = 8000.0d * Math.exp(-i / tau);
                double v = n / 1000.0d * env;
                out[start + i] = (short) Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, Math.round(v)));
            }
        }
        return out;
    }
}
