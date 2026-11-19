package com.aresstack.audio.profile;

import com.aresstack.audio.pipeline.AudioProcessingProfiles;

import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public final class AudioProcessingProfilesTest {

    @Test
    public void defaultSpeechProfileExposesAntiAliasFilterAndResampler() {
        AudioProcessingProfile profile = AudioProcessingProfiles.defaultSpeech();
        List<AudioBlockDefinition> blocks = profile.getBlocks();

        assertTrue(profile.isBuiltIn());
        assertEquals(AudioProcessingProfiles.DEFAULT_PROFILE_ID, profile.getId());
        assertEquals(AudioBlockType.CHANNEL_MIXER, blocks.get(0).getType());
        assertEquals(AudioBlockType.LOW_PASS, blocks.get(1).getType());
        assertEquals(AudioBlockType.RESAMPLER, blocks.get(2).getType());
        assertTrue(blocks.get(1).isEnabled());
        assertEquals("FIR_65", blocks.get(1).getParameter("implementation", ""));
        assertEquals(7200.0d, blocks.get(1).getDoubleParameter("cutoffHz", 0.0d), 0.0d);
        assertEquals("BALANCED", blocks.get(2).getParameter("quality", ""));
        assertFalse(blocks.get(2).getBooleanParameter("hiddenAntiAliasing", true));
    }
}
