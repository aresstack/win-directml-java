package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.dsp.Goertzel;
import com.aresstack.audio.dsp.NoiseProfile;
import com.aresstack.audio.profile.AudioBlockDefinition;
import com.aresstack.audio.profile.AudioBlockType;
import com.aresstack.audio.profile.AudioProcessingProfile;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/** The Noise Profiler and Adaptive Noise Suppression blocks integrate through the registry and work. */
public class NoiseSuppressionIntegrationTest {

    private static final int RATE = 16000;
    private static final PcmAudioFormat MONO = new PcmAudioFormat(RATE, 1, 16);
    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();

    @Test
    public void registryCreatesBothBlocks() {
        assertNotNull(registry.createProcessor(AudioBlockType.NOISE_PROFILER));
        assertNotNull(registry.createProcessor(AudioBlockType.ADAPTIVE_NOISE_SUPPRESSION));
        assertFalse(registry.descriptor(AudioBlockType.NOISE_PROFILER).getParameters().isEmpty());
        assertFalse(registry.descriptor(AudioBlockType.ADAPTIVE_NOISE_SUPPRESSION).getParameters().isEmpty());
        assertFalse("Noise Profiler is analysis-only",
                registry.descriptor(AudioBlockType.NOISE_PROFILER).getCapabilities().modifiesAudio());
    }

    @Test
    public void noiseProfilerLearnsAProfileAndLeavesAudioUnchanged() {
        AudioProcessingContext ctx = new AudioProcessingContext();
        short[] input = noise(24000, 4000);
        short[] out = registry.createProcessor(AudioBlockType.NOISE_PROFILER)
                .process(new AudioBuffer(input.clone(), MONO),
                        profiler("LEARN_FROM_SILENCE"), ctx).getSamples();
        assertArrayEquals("Noise Profiler must not change the audio", input, out);
        NoiseProfile profile = ctx.getNoiseProfile();
        assertNotNull(profile);
        assertTrue("profile has non-zero noise magnitude", profile.magnitudeAt(profile.getFftSize() / 4) > 0.0d);
    }

    @Test
    public void fixedProfileSuppressionReducesNoiseButKeepsALoudTone() {
        AudioProcessingContext ctx = new AudioProcessingContext();
        // Learn from a noise-only recording, then suppress a strong tone buried in the same noise.
        registry.createProcessor(AudioBlockType.NOISE_PROFILER)
                .process(new AudioBuffer(noise(24000, 4000), MONO), profiler("LEARN_FROM_SILENCE"), ctx);

        short[] noisy = add(tone(1000.0d, 24000, 9000), noise(24000, 4000));
        short[] out = registry.createProcessor(AudioBlockType.ADAPTIVE_NOISE_SUPPRESSION)
                .process(new AudioBuffer(noisy.clone(), MONO), suppression("USE_FIXED_PROFILE"), ctx)
                .getSamples();

        assertTrue("1 kHz tone preserved", powerAt(out, 1000.0d) > 0.6d * powerAt(noisy, 1000.0d));
        assertTrue("overall noise reduced", rms(out) < rms(noisy));
    }

    @Test
    public void automaticSuppressionReducesStationaryNoise() {
        short[] input = noise(48000, 4000);
        short[] out = registry.createProcessor(AudioBlockType.ADAPTIVE_NOISE_SUPPRESSION)
                .process(new AudioBuffer(input.clone(), MONO),
                        suppression("AUTOMATIC").withParameter("adaptationSpeed", "1.0")
                                .withParameter("speechProtection", "false"), new AudioProcessingContext())
                .getSamples();
        assertTrue("stationary noise attenuated: " + rms(out) + " vs " + rms(input),
                rms(out) < 0.85d * rms(input));
    }

    @Test
    public void disabledSuppressionLeavesTheSignalUnchanged() {
        AudioBlockDefinition disabled = suppression("AUTOMATIC").withEnabled(false);
        AudioProcessingProfile profile = new AudioProcessingProfile("p", "P", false, one(disabled));
        short[] samples = add(tone(1000.0d, 8000, 6000), noise(8000, 3000));
        short[] out = new AudioProfileProcessor()
                .process(new AudioBuffer(samples.clone(), MONO), profile).getSamples();
        assertArrayEquals(samples, out);
    }

    // ------------------------------------------------------------------ helpers

    private AudioBlockDefinition profiler(String mode) {
        return registry.defaultDefinition(AudioBlockType.NOISE_PROFILER, "np").withParameter("mode", mode);
    }

    private AudioBlockDefinition suppression(String mode) {
        return registry.defaultDefinition(AudioBlockType.ADAPTIVE_NOISE_SUPPRESSION, "ns")
                .withParameter("mode", mode);
    }

    private static List<AudioBlockDefinition> one(AudioBlockDefinition block) {
        List<AudioBlockDefinition> list = new ArrayList<AudioBlockDefinition>();
        list.add(block);
        return list;
    }

    private static short[] tone(double freq, int n, int amp) {
        short[] out = new short[n];
        for (int i = 0; i < n; i++) {
            out[i] = (short) Math.round(amp * Math.sin(2.0d * Math.PI * freq * i / RATE));
        }
        return out;
    }

    private static short[] noise(int n, int amp) {
        short[] out = new short[n];
        int state = 24681;
        for (int i = 0; i < n; i++) {
            state = state * 1103515245 + 12345;
            out[i] = (short) (((state >> 16) % (2 * amp)) - amp);
        }
        return out;
    }

    private static short[] add(short[] a, short[] b) {
        short[] out = new short[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = (short) Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, a[i] + b[i]));
        }
        return out;
    }

    private static double powerAt(short[] samples, double freq) {
        double[] mono = new double[samples.length];
        for (int i = 0; i < samples.length; i++) {
            mono[i] = samples[i];
        }
        int from = samples.length / 2;
        return Goertzel.power(mono, from, samples.length - from, RATE, freq);
    }

    private static double rms(short[] samples) {
        long sum = 0;
        int from = samples.length / 2;
        for (int i = from; i < samples.length; i++) {
            sum += (long) samples[i] * samples[i];
        }
        return Math.sqrt((double) sum / (samples.length - from));
    }
}
