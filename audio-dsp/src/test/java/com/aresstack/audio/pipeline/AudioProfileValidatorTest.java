package com.aresstack.audio.pipeline;

import com.aresstack.audio.profile.AudioBlockDefinition;
import com.aresstack.audio.profile.AudioBlockType;
import com.aresstack.audio.profile.AudioProcessingProfile;
import com.aresstack.audio.pipeline.AudioProcessingProfiles;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** The validator reports understandable problems without reordering the pipeline. */
public class AudioProfileValidatorTest {

    private final AudioProfileValidator validator = new AudioProfileValidator();
    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();

    @Test
    public void defaultProfileHasNoErrorsOrWarnings() {
        List<AudioProfileValidator.Message> messages = validator.validate(AudioProcessingProfiles.defaultSpeech());
        assertTrue("default profile is clean, but got: " + describe(messages), messages.isEmpty());
    }

    @Test
    public void invalidBandRangeIsAnError() {
        AudioBlockDefinition band = registry.defaultDefinition(AudioBlockType.BAND_PASS, "band")
                .withParameter("centerHz", "100")
                .withParameter("widthHz", "500"); // 100 - 250 < 0 → invalid
        AudioProcessingProfile profile = new AudioProcessingProfile("p", "P", false, one(band));
        assertTrue(hasSeverity(validator.validate(profile), AudioProfileValidator.Severity.ERROR));
    }

    @Test
    public void resamplerOutOfRangeIsAnError() {
        AudioBlockDefinition mixer = registry.defaultDefinition(AudioBlockType.CHANNEL_MIXER, "mix");
        AudioBlockDefinition resampler = registry.defaultDefinition(AudioBlockType.RESAMPLER, "res")
                .withParameter("targetRateHz", "1000"); // below the supported minimum
        List<AudioBlockDefinition> blocks = new ArrayList<AudioBlockDefinition>();
        blocks.add(mixer);
        blocks.add(resampler);
        AudioProcessingProfile profile = new AudioProcessingProfile("p", "P", false, blocks);
        assertTrue(hasSeverity(validator.validate(profile), AudioProfileValidator.Severity.ERROR));
    }

    @Test
    public void resamplerWithoutChannelMixerWarns() {
        AudioBlockDefinition resampler = registry.defaultDefinition(AudioBlockType.RESAMPLER, "res");
        AudioProcessingProfile profile = new AudioProcessingProfile("p", "P", false, one(resampler));
        List<AudioProfileValidator.Message> messages = validator.validate(profile);
        assertTrue(hasSeverity(messages, AudioProfileValidator.Severity.WARNING));
        assertFalse(hasSeverity(messages, AudioProfileValidator.Severity.ERROR));
    }

    private static List<AudioBlockDefinition> one(AudioBlockDefinition block) {
        List<AudioBlockDefinition> blocks = new ArrayList<AudioBlockDefinition>();
        blocks.add(block);
        return blocks;
    }

    private static boolean hasSeverity(List<AudioProfileValidator.Message> messages,
                                       AudioProfileValidator.Severity severity) {
        for (int i = 0; i < messages.size(); i++) {
            if (messages.get(i).getSeverity() == severity) {
                return true;
            }
        }
        return false;
    }

    private static String describe(List<AudioProfileValidator.Message> messages) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < messages.size(); i++) {
            builder.append(messages.get(i).getSeverity()).append(": ").append(messages.get(i).getText()).append("; ");
        }
        return builder.toString();
    }
}
