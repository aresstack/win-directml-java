package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.dsp.Goertzel;
import com.aresstack.audio.profile.AudioBlockDefinition;
import com.aresstack.audio.profile.AudioBlockType;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/** The channel-strip Equalizer shapes tone, applies a master gain and offers a safe loudness stage. */
public class EqualizerBlockTest {

    private static final int RATE = 16000;
    private static final PcmAudioFormat MONO = new PcmAudioFormat(RATE, 1, 16);
    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();

    @Test
    public void isRegisteredAsAnAudioEffect() {
        assertNotNull(registry.createProcessor(AudioBlockType.EQUALIZER));
        assertFalse(registry.descriptor(AudioBlockType.EQUALIZER).getParameters().isEmpty());
        assertTrue(registry.descriptor(AudioBlockType.EQUALIZER).getCapabilities().modifiesAudio());
    }

    @Test
    public void flatSettingsLeaveTheAudioUnchanged() {
        short[] samples = tone(500.0d, 8000, 6000);
        short[] original = samples.clone();
        process(registry.defaultDefinition(AudioBlockType.EQUALIZER, "eq"), samples); // all gains 0, loudness off
        assertArrayEquals(original, samples);
    }

    @Test
    public void masterGainRaisesTheLevel() {
        short[] samples = tone(500.0d, 8000, 5000);
        double before = rms(samples);
        process(registry.defaultDefinition(AudioBlockType.EQUALIZER, "eq").withParameter("gainDb", "6"), samples);
        assertTrue("master gain +6 dB roughly doubles the level: " + before + " -> " + rms(samples),
                rms(samples) > before * 1.8d);
    }

    @Test
    public void loudnessStageStaysUnderTheCeilingInsteadOfHardClipping() {
        short[] samples = tone(500.0d, 8000, 22000);
        AudioBlockDefinition eq = registry.defaultDefinition(AudioBlockType.EQUALIZER, "eq")
                .withParameter("loudness", "true")
                .withParameter("loudnessDriveDb", "12")
                .withParameter("peakCeilingDb", "-0.5");
        process(eq, samples);
        double ceiling = Math.pow(10.0d, -0.5d / 20.0d) * 32768.0d;
        double peak = peak(samples);
        assertTrue("soft saturation keeps peaks under the ceiling: " + peak + " <= " + ceiling,
                peak <= ceiling + 1.0d);
        assertTrue("no hard clipping to full scale", peak < 32767.0d);
    }

    @Test
    public void lowShelfBoostRaisesLowFrequencyEnergy() {
        short[] input = tone(100.0d, 8000, 6000);
        double before = powerAt(input, 100.0d);
        short[] work = input.clone();
        process(registry.defaultDefinition(AudioBlockType.EQUALIZER, "eq")
                .withParameter("lowShelfHz", "120").withParameter("lowShelfGainDb", "12"), work);
        assertTrue("low shelf boosts the low tone: " + before + " -> " + powerAt(work, 100.0d),
                powerAt(work, 100.0d) > before * 2.0d);
    }

    private void process(AudioBlockDefinition block, short[] samples) {
        registry.createProcessor(AudioBlockType.EQUALIZER)
                .process(new AudioBuffer(samples, MONO), block, new AudioProcessingContext());
    }

    private static short[] tone(double freq, int n, int amp) {
        short[] out = new short[n];
        for (int i = 0; i < n; i++) {
            out[i] = (short) Math.round(amp * Math.sin(2.0d * Math.PI * freq * i / RATE));
        }
        return out;
    }

    private static double rms(short[] s) {
        long sum = 0;
        int from = s.length / 2;
        for (int i = from; i < s.length; i++) {
            sum += (long) s[i] * s[i];
        }
        return Math.sqrt((double) sum / (s.length - from));
    }

    private static double peak(short[] s) {
        double p = 0.0d;
        for (short v : s) {
            p = Math.max(p, Math.abs(v));
        }
        return p;
    }

    private static double powerAt(short[] s, double freq) {
        double[] mono = new double[s.length];
        for (int i = 0; i < s.length; i++) {
            mono[i] = s[i];
        }
        int from = s.length / 2;
        return Goertzel.power(mono, from, s.length - from, RATE, freq);
    }
}
