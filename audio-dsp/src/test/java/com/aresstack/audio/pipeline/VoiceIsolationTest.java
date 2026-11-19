package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.profile.AudioBlockDefinition;
import com.aresstack.audio.profile.AudioBlockType;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/** Slice 13D: Voice Isolation keeps the dominant centred voice, reduces lateral content, degrades safely. */
public class VoiceIsolationTest {

    private static final int RATE = 16000;
    private static final PcmAudioFormat STEREO = new PcmAudioFormat(RATE, 2, 16);
    private static final PcmAudioFormat MONO = new PcmAudioFormat(RATE, 1, 16);
    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();

    @Test
    public void isRegistered() {
        assertNotNull(registry.createProcessor(AudioBlockType.VOICE_ISOLATION));
    }

    @Test
    public void pureJavaCenterReducesLateralInterferers() {
        short[] voice = tone(300.0d, 4000, 8000);       // centred: L = R
        short[] interferer = tone(1500.0d, 4000, 6000); // lateral: L = -R
        short[] left = new short[voice.length];
        short[] right = new short[voice.length];
        for (int i = 0; i < voice.length; i++) {
            left[i] = clamp(voice[i] + interferer[i]);
            right[i] = clamp(voice[i] - interferer[i]);
        }
        AudioBuffer out = run(registry.defaultDefinition(AudioBlockType.VOICE_ISOLATION, "v")
                .withParameter("backend", "PURE_JAVA_CENTER").withParameter("strength", "1"),
                interleave(left, right), STEREO);
        // With side fully removed, both channels become the centred voice: L == R.
        short[] s = out.getSamples();
        for (int f = 0; f < s.length / 2; f++) {
            assertTrue("lateral removed -> L==R", Math.abs(s[2 * f] - s[2 * f + 1]) <= 1);
        }
    }

    @Test
    public void neuralBackendPassesThrough() {
        short[] stereo = interleave(tone(300.0d, 2000, 8000), tone(1500.0d, 2000, 6000));
        AudioBuffer out = run(registry.defaultDefinition(AudioBlockType.VOICE_ISOLATION, "v")
                .withParameter("backend", "NEURAL"), stereo, STEREO);
        assertArrayEquals(stereo, out.getSamples());
    }

    @Test
    public void monoSourcePassesThrough() {
        short[] mono = tone(300.0d, 2000, 8000);
        AudioBuffer out = run(registry.defaultDefinition(AudioBlockType.VOICE_ISOLATION, "v")
                .withParameter("backend", "PURE_JAVA_CENTER"), mono, MONO);
        assertArrayEquals(mono, out.getSamples());
    }

    private AudioBuffer run(AudioBlockDefinition block, short[] samples, PcmAudioFormat format) {
        return registry.createProcessor(AudioBlockType.VOICE_ISOLATION)
                .process(new AudioBuffer(samples, format), block, new AudioProcessingContext());
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

    private static short clamp(int v) {
        return (short) Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, v));
    }
}
