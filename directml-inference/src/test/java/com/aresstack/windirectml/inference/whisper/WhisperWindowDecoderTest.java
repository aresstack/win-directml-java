package com.aresstack.windirectml.inference.whisper;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WhisperWindowDecoderTest {

    /** Output reported from Space Harvester's chat dictation with directml-inference 0.3.1. */
    private static final String REPORTED_LOOP = "Ich dachte mir, du kannst mal im Internet nach weiter "
            + "unterteis so" + " ...".repeat(60) + " ZARB-K-Suchen und mir alle detailierte "
            + "Zusammenfassungen erstellen." + " den".repeat(80);

    @Test
    void reportedDictationLoopFailsTheCompressionCheck() {
        assertTrue(WhisperWindowDecoder.compressionRatio(REPORTED_LOOP)
                > WhisperDecodingOptions.defaults().compressionRatioThreshold());
    }

    @Test
    void ordinaryGermanSpeechPassesTheCompressionCheck() {
        String speech = "Ich dachte mir, du kannst mal im Internet nach weiteren Unterlagen suchen "
                + "und mir detaillierte Zusammenfassungen erstellen. Danach schicke die "
                + "Zusammenfassung an das gesamte Team.";

        assertTrue(WhisperWindowDecoder.compressionRatio(speech)
                < WhisperDecodingOptions.defaults().compressionRatioThreshold());
        assertEquals(0.0, WhisperWindowDecoder.compressionRatio(""));
    }

    @Test
    void compressionRatioEqualsTheZlibValueOfTheReferenceImplementation() {
        // Python: len(t) / len(zlib.compress(t)) == 123 / 101 for this sentence.
        String sentence = "Ich dachte mir, du kannst mal im Internet nach weiteren Unterlagen suchen "
                + "und mir detaillierte Zusammenfassungen erstellen.";

        assertEquals(123.0 / 101.0, WhisperWindowDecoder.compressionRatio(sentence), 1e-12);
    }

    @Test
    void treatsPunctuationOnlyOutputAsNoSpeech() {
        assertFalse(WhisperWindowDecoder.containsWord(" ... "));
        assertFalse(WhisperWindowDecoder.containsWord(""));
        assertTrue(WhisperWindowDecoder.containsWord("Hallo."));
        assertTrue(WhisperWindowDecoder.containsWord("42"));
    }

    @Test
    void prefersTheMostProbableNonRepetitiveAttemptWhenEveryTemperatureFails() {
        WhisperWindowDecoder.Attempt greedyLoop = attempt(0.0, -0.4, 3.1, true);
        WhisperWindowDecoder.Attempt plausible = attempt(0.2, -1.2, 1.4, false);
        WhisperWindowDecoder.Attempt noisy = attempt(1.0, -4.8, 0.9, false);

        assertSame(plausible, WhisperWindowDecoder.mostProbable(
                Arrays.asList(greedyLoop, plausible, noisy), WhisperDecodingOptions.defaults()));
    }

    @Test
    void fallsBackToTheMostProbableAttemptWhenAllAreRepetitive() {
        WhisperWindowDecoder.Attempt first = attempt(0.0, -0.6, 3.0, true);
        WhisperWindowDecoder.Attempt second = attempt(0.2, -2.0, 2.9, true);

        assertSame(first, WhisperWindowDecoder.mostProbable(
                Arrays.asList(first, second), WhisperDecodingOptions.defaults()));
    }

    @Test
    void defaultsFollowTheReferenceSafeguards() {
        WhisperDecodingOptions defaults = WhisperDecodingOptions.defaults();

        assertEquals(6, defaults.temperatures().length);
        assertEquals(0.0, defaults.temperatures()[0]);
        assertEquals(1.0, defaults.temperatures()[5]);
        assertEquals(2.4, defaults.compressionRatioThreshold());
        assertEquals(-1.0, defaults.logProbabilityThreshold());
        assertEquals(0.6, defaults.noSpeechThreshold());
        assertEquals(224, defaults.maximumTokensPerWindow());
        assertTrue(defaults.fullWindowEncoderInput());
        assertEquals("", defaults.initialPrompt(), "A prompt can leak into silent windows; it stays opt-in");
        assertEquals("Confluence", defaults.withInitialPrompt("  Confluence ").initialPrompt());
        assertEquals("Confluence", defaults.withInitialPrompt("Confluence").withMaximumTokensPerWindow(100)
                .initialPrompt());
        assertEquals("de", defaults.language());
        assertEquals("auto", defaults.withLanguage(" AUTO ").language());
        assertEquals("en", defaults.withLanguage("en").withInitialPrompt("x").language());
        assertThrows(IllegalArgumentException.class, () -> defaults.withLanguage("german"));
        assertThrows(IllegalArgumentException.class, defaults::withTemperatures);
        assertThrows(IllegalArgumentException.class, () -> defaults.withMaximumTokensPerWindow(0));
    }

    private static WhisperWindowDecoder.Attempt attempt(double temperature, double logProbability,
                                                        double compressionRatio, boolean repetitive) {
        return new WhisperWindowDecoder.Attempt(Collections.emptyList(), "text", temperature,
                logProbability, compressionRatio, 0.1, repetitive);
    }
}
