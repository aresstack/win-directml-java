package com.aresstack.audio.pipeline;

import com.aresstack.audio.profile.AudioBlockDefinition;
import com.aresstack.audio.profile.AudioBlockType;
import com.aresstack.audio.profile.AudioProcessingProfile;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** Validator ordering and parameter rules for the Expander and the Silence Trimmer. */
public class ExpanderSilenceTrimmerValidationTest {

    private final AudioProfileValidator validator = new AudioProfileValidator();
    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();

    @Test
    public void vadThenTrimmerIsValid() {
        AudioProfileValidationResult result = validator.validateResult(profile(vad(), trimmer()));
        assertFalse(result.hasErrors());
    }

    @Test
    public void trimmerWithoutVadIsAnError() {
        assertTrue(validator.validateResult(profile(trimmer())).hasErrors());
    }

    @Test
    public void trimmerBeforeVadIsAnError() {
        assertTrue(validator.validateResult(profile(trimmer(), vad())).hasErrors());
    }

    @Test
    public void aTimeBaseChangingBlockBetweenVadAndTrimmerIsAnError() {
        AudioBlockDefinition resampler = registry.defaultDefinition(AudioBlockType.RESAMPLER, "res");
        AudioProfileValidationResult result = validator.validateResult(profile(vad(), resampler, trimmer()));
        assertTrue(result.hasErrors());
    }

    @Test
    public void resamplerBeforeVadIsFine() {
        AudioBlockDefinition mixer = registry.defaultDefinition(AudioBlockType.CHANNEL_MIXER, "mix");
        AudioBlockDefinition resampler = registry.defaultDefinition(AudioBlockType.RESAMPLER, "res");
        AudioProfileValidationResult result =
                validator.validateResult(profile(mixer, resampler, vad(), trimmer()));
        assertFalse("resampler → VAD → trimmer is the recommended order", result.hasErrors());
    }

    @Test
    public void multipleTrimmersWarnOnTheSecond() {
        AudioProfileValidationResult result = validator.validateResult(profile(vad(), trimmer(), trimmer("t2")));
        assertTrue(result.hasWarnings());
    }

    @Test
    public void speechProtectionWithoutVadIsAWarningNotError() {
        AudioBlockDefinition expander = registry.defaultDefinition(AudioBlockType.EXPANDER, "exp")
                .withParameter("speechProtection", "true");
        AudioProfileValidationResult result = validator.validateResult(profile(expander));
        assertTrue(result.hasWarnings());
        assertFalse(result.hasErrors());
    }

    @Test
    public void expanderRatioBelowOneIsAnError() {
        AudioBlockDefinition expander = registry.defaultDefinition(AudioBlockType.EXPANDER, "exp")
                .withParameter("ratio", "0.5");
        AudioProfileValidationResult result = validator.validateResult(profile(expander));
        assertTrue(result.hasErrors());
        assertEquals("ratio", result.issuesForBlock("exp").get(0).getParameterKey());
    }

    @Test
    public void expanderNegativeMaxAttenuationIsAnError() {
        AudioBlockDefinition expander = registry.defaultDefinition(AudioBlockType.EXPANDER, "exp")
                .withParameter("maxAttenuationDb", "-3");
        assertTrue(validator.validateResult(profile(expander)).hasErrors());
    }

    @Test
    public void negativePreRollIsAnError() {
        AudioBlockDefinition trimmer = trimmer().withParameter("preRollMs", "-10");
        assertTrue(validator.validateResult(profile(vad(), trimmer)).hasErrors());
    }

    // ------------------------------------------------------------------ helpers

    private AudioBlockDefinition vad() {
        return registry.defaultDefinition(AudioBlockType.VOICE_ACTIVITY_DETECTION, "vad");
    }

    private AudioBlockDefinition trimmer() {
        return trimmer("trim");
    }

    private AudioBlockDefinition trimmer(String id) {
        return registry.defaultDefinition(AudioBlockType.SILENCE_TRIMMER, id);
    }

    private static AudioProcessingProfile profile(AudioBlockDefinition... blocks) {
        List<AudioBlockDefinition> list = new ArrayList<AudioBlockDefinition>();
        for (AudioBlockDefinition block : blocks) {
            list.add(block);
        }
        return new AudioProcessingProfile("p", "P", false, list);
    }
}
