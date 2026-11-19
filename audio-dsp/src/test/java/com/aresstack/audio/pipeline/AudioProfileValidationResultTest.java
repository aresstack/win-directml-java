package com.aresstack.audio.pipeline;

import com.aresstack.audio.profile.AudioBlockDefinition;
import com.aresstack.audio.profile.AudioBlockType;
import com.aresstack.audio.profile.AudioProcessingProfile;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/** The rich validation result carries block id, parameter key and severity for the editor to surface. */
public class AudioProfileValidationResultTest {

    private final AudioProfileValidator validator = new AudioProfileValidator();
    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();

    @Test
    public void validProfileHasNoIssues() {
        AudioProfileValidationResult result = validator.validateResult(AudioProcessingProfiles.defaultSpeech());
        assertFalse(result.hasErrors());
        assertFalse(result.hasWarnings());
        assertTrue(result.getIssues().isEmpty());
    }

    @Test
    public void invalidEqualizerYieldsAnErrorWithBlockAndParameterKey() {
        AudioBlockDefinition eq = registry.defaultDefinition(AudioBlockType.PARAMETRIC_EQ, "eq1")
                .withParameter("centerHz", "0");
        AudioProfileValidationResult result = validator.validateResult(profile(eq));
        assertTrue(result.hasErrors());
        AudioProfileValidationIssue issue = firstFor(result, "eq1");
        assertNotNull(issue);
        assertEquals(AudioValidationSeverity.ERROR, issue.getSeverity());
        assertEquals(AudioBlockType.PARAMETRIC_EQ, issue.getBlockType());
        assertEquals("centerHz", issue.getParameterKey());
    }

    @Test
    public void invalidVadYieldsAnErrorWithTheOffendingParameterKey() {
        AudioBlockDefinition vad = registry.defaultDefinition(AudioBlockType.VOICE_ACTIVITY_DETECTION, "vad1")
                .withParameter("minSpeechProbability", "2.0"); // out of 0..1
        AudioProfileValidationResult result = validator.validateResult(profile(vad));
        AudioProfileValidationIssue issue = firstFor(result, "vad1");
        assertNotNull(issue);
        assertEquals(AudioValidationSeverity.ERROR, issue.getSeverity());
        assertEquals("minSpeechProbability", issue.getParameterKey());
    }

    @Test
    public void errorsAndWarningsAreSeparated() {
        AudioBlockDefinition gain = registry.defaultDefinition(AudioBlockType.GAIN, "g")
                .withParameter("gainDb", "60"); // extreme → warning
        AudioBlockDefinition eq = registry.defaultDefinition(AudioBlockType.PARAMETRIC_EQ, "eq")
                .withParameter("q", "0"); // invalid → error
        AudioProfileValidationResult result = validator.validateResult(profile(gain, eq));
        assertTrue(result.hasErrors());
        assertTrue(result.hasWarnings());
        assertEquals(1, result.errorCount());
        assertEquals(1, result.warningCount());
    }

    @Test
    public void aDisabledInvalidBlockIsDowngradedToWarningAndDoesNotBlock() {
        AudioBlockDefinition eq = registry.defaultDefinition(AudioBlockType.PARAMETRIC_EQ, "eqOff")
                .withParameter("centerHz", "0").withEnabled(false);
        AudioProfileValidationResult result = validator.validateResult(profile(eq));
        assertFalse("a bypassed block must not block processing", result.hasErrors());
        assertTrue("but its problem stays visible as a warning", result.hasWarnings());
        assertEquals(AudioValidationSeverity.WARNING, firstFor(result, "eqOff").getSeverity());
    }

    @Test
    public void nonParseableNumberIsReportedGenerically() {
        AudioBlockDefinition eq = registry.defaultDefinition(AudioBlockType.PARAMETRIC_EQ, "eqBad")
                .withParameter("q", "abc");
        AudioProfileValidationResult result = validator.validateResult(profile(eq));
        assertTrue(result.hasErrors());
        assertEquals("q", firstFor(result, "eqBad").getParameterKey());
    }

    private AudioProcessingProfile profile(AudioBlockDefinition... blocks) {
        List<AudioBlockDefinition> list = new ArrayList<AudioBlockDefinition>();
        for (AudioBlockDefinition block : blocks) {
            list.add(block);
        }
        return new AudioProcessingProfile("p", "P", false, list);
    }

    private static AudioProfileValidationIssue firstFor(AudioProfileValidationResult result, String blockId) {
        List<AudioProfileValidationIssue> issues = result.issuesForBlock(blockId);
        return issues.isEmpty() ? null : issues.get(0);
    }
}
