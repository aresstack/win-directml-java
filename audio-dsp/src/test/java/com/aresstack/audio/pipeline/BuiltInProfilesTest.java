package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.profile.AudioBlockDefinition;
import com.aresstack.audio.profile.AudioBlockType;
import com.aresstack.audio.profile.AudioProcessingProfile;

import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** The built-in profiles (Off, Default speech, Crystal voice) are correct and protected. */
public class BuiltInProfilesTest {

    @Test
    public void thereAreThreeProtectedBuiltIns() {
        List<AudioProcessingProfile> builtIns = AudioProcessingProfiles.builtIns();
        assertEquals(3, builtIns.size());
        for (AudioProcessingProfile profile : builtIns) {
            assertTrue("built-in flag for " + profile.getName(), profile.isBuiltIn());
            assertTrue(AudioProcessingProfiles.isBuiltInId(profile.getId()));
        }
        assertFalse(AudioProcessingProfiles.isBuiltInId("something-else"));
    }

    @Test
    public void offProfileHasNoBlocks() {
        AudioProcessingProfile off = AudioProcessingProfiles.off();
        assertEquals(AudioProcessingProfiles.OFF_PROFILE_ID, off.getId());
        assertTrue(off.getBlocks().isEmpty());
    }

    @Test
    public void crystalVoiceMatchesTheClearSpeechChain() {
        AudioProcessingProfile cv = AudioProcessingProfiles.crystalVoice();
        assertEquals("Crystal voice", cv.getName());
        List<AudioBlockDefinition> blocks = cv.getBlocks();
        assertEquals(13, blocks.size());

        AudioBlockDefinition ans = find(blocks, AudioBlockType.ADAPTIVE_NOISE_SUPPRESSION);
        assertEquals("16", ans.getParameter("maxAttenuationDb", ""));

        AudioBlockDefinition eq = blocks.get(blocks.size() - 1);
        assertEquals(AudioBlockType.EQUALIZER, eq.getType());
        assertFalse("the equalizer ships disabled on purpose", eq.isEnabled());
        assertEquals("5", eq.getParameter("gainDb", ""));
        assertEquals("true", eq.getParameter("loudness", ""));
    }

    @Test
    public void crystalVoiceIsAValidProfile() {
        AudioProcessingProfile cv = AudioProcessingProfiles.crystalVoice();
        assertFalse(new AudioProfileValidator()
                .validateResult(cv, new PcmAudioFormat(48000, 1, 16)).hasErrors());
    }

    private static AudioBlockDefinition find(List<AudioBlockDefinition> blocks, AudioBlockType type) {
        for (AudioBlockDefinition block : blocks) {
            if (block.getType() == type) {
                return block;
            }
        }
        throw new AssertionError("missing block " + type);
    }
}
