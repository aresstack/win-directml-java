package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.dsp.Goertzel;
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

/** The FFT (STFT) block variants integrate through the registry and behave as intended. */
public class SpectralBlocksIntegrationTest {

    private static final int RATE = 16000;
    private static final PcmAudioFormat MONO = new PcmAudioFormat(RATE, 1, 16);
    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();

    @Test
    public void registryCreatesAllFourFftBlocks() {
        for (AudioBlockType type : new AudioBlockType[]{AudioBlockType.DE_ESSER_FFT,
                AudioBlockType.ADAPTIVE_HUM_REMOVAL_FFT, AudioBlockType.PLOSIVE_REDUCTION_FFT,
                AudioBlockType.BREATH_REDUCTION_FFT}) {
            assertNotNull(registry.createProcessor(type));
            assertFalse(registry.descriptor(type).getParameters().isEmpty());
            assertTrue("spectral blocks require framing",
                    registry.descriptor(type).getCapabilities().requiresFraming());
        }
    }

    @Test
    public void fftDeEsserReducesTheSibilanceBandButNotLowTones() {
        AudioBlockDefinition block = registry.defaultDefinition(AudioBlockType.DE_ESSER_FFT, "d")
                .withParameter("thresholdDb", "-40").withParameter("reductionDb", "18");
        double sib = ratio(process(block, tone(6500.0d, RATE, 8000)), tone(6500.0d, RATE, 8000));
        double low = ratio(process(block, tone(300.0d, RATE, 8000)), tone(300.0d, RATE, 8000));
        assertTrue("sibilance reduced: " + sib, sib < 0.8d);
        assertTrue("low tone kept: " + low, low > 0.85d);
    }

    @Test
    public void fftHumRemovalReducesTheMainsToneButKeepsSpeechRangeContent() {
        short[] input = mix(1000.0d, 6000, 50.0d, 4000);
        short[] out = process(registry.defaultDefinition(AudioBlockType.ADAPTIVE_HUM_REMOVAL_FFT, "h")
                .withParameter("harmonics", "1").withParameter("maxAttenuationDb", "30"), input);
        assertTrue("50 Hz reduced", powerAt(out, 50.0d) < 0.5d * powerAt(input, 50.0d));
        assertTrue("1 kHz preserved", powerAt(out, 1000.0d) > 0.7d * powerAt(input, 1000.0d));
    }

    @Test
    public void fftBreathRemovalAttenuatesNoiseMoreThanATone() {
        AudioBlockDefinition block = registry.defaultDefinition(AudioBlockType.BREATH_REDUCTION_FFT, "b")
                .withParameter("sensitivity", "0.9").withParameter("maxAttenuationDb", "24");
        double noise = ratio(process(block, noise(RATE)), noise(RATE));
        double tone = ratio(process(block, tone(1000.0d, RATE, 6000)), tone(1000.0d, RATE, 6000));
        assertTrue("noise attenuated more than a tone: noise=" + noise + " tone=" + tone, noise < tone);
        assertTrue("noise is attenuated", noise < 0.9d);
    }

    @Test
    public void defaultSpeechContainsNoFftBlockAndStaysBitIdentical() {
        AudioProcessingProfile def = AudioProcessingProfiles.defaultSpeech();
        for (AudioBlockDefinition block : def.getBlocks()) {
            assertFalse(block.getType().name().endsWith("_FFT"));
        }
        AudioBuffer in = new AudioBuffer(tone(500.0d, 9600, 6000), new PcmAudioFormat(48000, 2, 16));
        short[] a = new AudioProfileProcessor().process(copy(in), def).getSamples();
        short[] b = new AudioProfileProcessor().process(copy(in), def).getSamples();
        assertArrayEquals(a, b);
    }

    @Test
    public void aDisabledFftBlockLeavesTheSignalUnchanged() {
        AudioBlockDefinition disabled = registry.defaultDefinition(AudioBlockType.DE_ESSER_FFT, "d")
                .withParameter("reductionDb", "24").withEnabled(false);
        AudioProcessingProfile profile = new AudioProcessingProfile("p", "P", false, one(disabled));
        short[] samples = tone(6500.0d, 8000, 9000);
        short[] out = new AudioProfileProcessor()
                .process(new AudioBuffer(samples.clone(), MONO), profile).getSamples();
        assertArrayEquals(samples, out);
    }

    // ------------------------------------------------------------------ helpers

    private short[] process(AudioBlockDefinition block, short[] input) {
        AudioBuffer out = registry.createProcessor(block.getType())
                .process(new AudioBuffer(input.clone(), MONO), block, new AudioProcessingContext());
        return out.getSamples();
    }

    private static double ratio(short[] out, short[] in) {
        return rms(out) / rms(in);
    }

    private static AudioBuffer copy(AudioBuffer b) {
        return new AudioBuffer(b.getSamples().clone(), b.getFormat());
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

    private static short[] mix(double f1, int a1, double f2, int a2) {
        short[] out = new short[RATE];
        for (int i = 0; i < RATE; i++) {
            double v = a1 * Math.sin(2.0d * Math.PI * f1 * i / RATE) + a2 * Math.sin(2.0d * Math.PI * f2 * i / RATE);
            out[i] = (short) Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, Math.round(v)));
        }
        return out;
    }

    private static short[] noise(int n) {
        short[] out = new short[n];
        int state = 13579;
        for (int i = 0; i < n; i++) {
            state = state * 1103515245 + 12345;
            out[i] = (short) ((state >> 16) % 6000);
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
