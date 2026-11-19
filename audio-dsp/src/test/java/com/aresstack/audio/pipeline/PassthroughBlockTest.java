package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.profile.AudioBlockDefinition;
import com.aresstack.audio.profile.AudioBlockType;
import com.aresstack.audio.profile.AudioProcessingProfile;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/** The neutral "---" passthrough block is registered, does nothing and is always valid. */
public class PassthroughBlockTest {

    private static final PcmAudioFormat MONO = new PcmAudioFormat(16000, 1, 16);
    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();

    @Test
    public void passthroughIsRegisteredAsANeutralPlaceholder() {
        assertNotNull(registry.createProcessor(AudioBlockType.PASSTHROUGH));
        assertEquals("---", AudioBlockType.PASSTHROUGH.getDisplayName());
        assertTrue(registry.descriptor(AudioBlockType.PASSTHROUGH).getParameters().isEmpty());
        assertFalse(registry.descriptor(AudioBlockType.PASSTHROUGH).getCapabilities().modifiesAudio());
    }

    @Test
    public void passthroughLeavesAudioBitIdentical() {
        short[] samples = tone(400.0d, 8000, 6000);
        short[] original = samples.clone();
        AudioProcessingProfile profile = new AudioProcessingProfile("p", "p", false,
                one(registry.defaultDefinition(AudioBlockType.PASSTHROUGH, "b")));
        short[] out = new AudioProfileProcessor().process(new AudioBuffer(samples, MONO), profile).getSamples();
        assertArrayEquals(original, out);
    }

    @Test
    public void passthroughHasNoValidationErrors() {
        AudioProcessingProfile profile = new AudioProcessingProfile("p", "p", false,
                one(registry.defaultDefinition(AudioBlockType.PASSTHROUGH, "b")));
        assertFalse(new AudioProfileValidator().validateResult(profile, MONO).hasErrors());
    }

    private static List<AudioBlockDefinition> one(AudioBlockDefinition block) {
        List<AudioBlockDefinition> list = new ArrayList<AudioBlockDefinition>();
        list.add(block);
        return list;
    }

    private static short[] tone(double freq, int n, int amp) {
        short[] out = new short[n];
        for (int i = 0; i < n; i++) {
            out[i] = (short) Math.round(amp * Math.sin(2.0d * Math.PI * freq * i / 16000.0d));
        }
        return out;
    }
}
