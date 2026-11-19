package com.aresstack.windirectml.whisper.audio;

import com.aresstack.windirectml.inference.whisper.TranscriptionResult;
import com.aresstack.windirectml.inference.whisper.TranscriptionSegment;
import com.aresstack.windirectml.inference.whisper.WhisperDecodingOptions;
import com.aresstack.windirectml.inference.whisper.WhisperTranscriber;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static com.aresstack.windirectml.whisper.audio.DictationScenarios.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Guard the 0.3.1 dictation failure: Whisper Tiny produced "den den den ..." and whole
 * repeated sentences. Run real-model scenarios through the default pipeline and through
 * the decoder alone with loud noise and long pauses, as if the VAD had failed.
 * Skip when the Whisper Tiny checkpoint is not present under model/whisper-tiny.
 */
class WhisperDictationRegressionTest {
    private static final double MAXIMUM_WORD_ERROR_RATE = 0.15;
    private static WhisperTranscriber transcriber;

    @BeforeAll
    static void loadModel() throws Exception {
        Assumptions.assumeTrue(modelAvailable(), "Whisper Tiny checkpoint not available");
        transcriber = new WhisperTranscriber(MODEL);
    }

    @Test
    void defaultPipelineTranscribesDictationWithoutRepetitionLoops() throws Exception {
        AskAiWhisperPipeline pipeline = new AskAiWhisperPipeline();
        for (Scenario scenario : all()) {
            long started = System.nanoTime();
            List<TranscriptionResult> progress = new ArrayList<>();
            TranscriptionResult result = pipeline.transcribe(source(scenario.audio), transcriber, progress::add);
            report("pipeline", scenario.name, scenario.seconds(), started, scenario.reference, result.getText());

            assertQuality(scenario.name, scenario.reference, result.getText());
            for (TranscriptionSegment segment : result.getSegments()) {
                assertTrue(segment.getEndMillis() <= scenario.audio.length * 1000L / SAMPLE_RATE,
                        scenario.name + ": segment exceeds the recording");
            }
        }
    }

    @Test
    void decoderWithoutVadSurvivesLoudNoiseAndLongPauses() throws Exception {
        Object[][] cases = {
                {"word + 12 s noise", WORD, noisy(silence(0.5), speech("de-word.wav"), silence(12))},
                {"sentence + 15 s noise", SENTENCE, noisy(speech("de-sentence.wav"), silence(15))},
                {"pauses of 6 s and 8 s", PAUSE_A + " " + PAUSE_B + " " + PAUSE_A,
                        noisy(speech("de-pause-a.wav"), silence(6), speech("de-pause-b.wav"),
                                silence(8), speech("de-pause-a.wav"))},
                {"noise only", "", noisy(silence(20))},
        };
        for (Object[] scenario : cases) {
            float[] audio = (float[]) scenario[2];
            long started = System.nanoTime();
            String text = transcriber.transcribe(audio, WhisperDecodingOptions.defaults(), ignored -> { })
                    .getText();
            report("decoder", (String) scenario[0], audio.length / (double) SAMPLE_RATE, started,
                    (String) scenario[1], text);

            assertQuality((String) scenario[0], (String) scenario[1], text);
        }
    }

    /**
     * A word at -28 dB is missed by the energy VAD and must be recovered by the whole-recording
     * fallback. After loud speech Whisper Tiny itself needs about -16 dB; below that the decoder
     * drops the word with or without VAD, so the VAD must never lose more than the decoder alone.
     */
    @Test
    void recognizesQuietWordsAndNeverLosesMoreThanTheDecoderAlone() throws Exception {
        AskAiWhisperPipeline pipeline = new AskAiWhisperPipeline();
        Object[][] cases = {
                {"word at -28 dB alone", WORD, microphone(silence(1.0), scaled(speech("de-word.wav"), 0.04f), silence(1.0))},
                {"word at -16 dB after loud speech", PAUSE_A + " " + WORD, microphone(silence(0.5), speech("de-pause-a.wav"),
                        silence(4.0), scaled(speech("de-word.wav"), 0.15f), silence(1.0))},
        };
        for (Object[] scenario : cases) {
            float[] audio = (float[]) scenario[2];
            long started = System.nanoTime();
            String withVad = pipeline.transcribePlain(source(audio), transcriber);
            report("pipeline", (String) scenario[0], audio.length / (double) SAMPLE_RATE, started,
                    (String) scenario[1], withVad);
            String withoutVad = transcriber.transcribe(audio, WhisperDecodingOptions.defaults(), ignored -> { })
                    .getText();

            assertTrue(withVad.toLowerCase(java.util.Locale.ROOT).contains("hallo"),
                    scenario[0] + ": the quiet word was lost: " + withVad);
            assertTrue(wordCount(withVad) >= wordCount(withoutVad),
                    scenario[0] + ": the VAD removed speech: " + withVad + " | without VAD: " + withoutVad);
        }
    }

    @Test
    void transcribesEnglishWhenSetAndDetectsTheLanguageAutomatically() throws Exception {
        String english = "Please search the knowledge base and summarize all pages about travel expenses.";
        float[] englishAudio = microphone(silence(0.5), speech("en-sentence.wav"), silence(0.5));
        float[] germanAudio = microphone(silence(0.5), speech("de-sentence.wav"), silence(0.5));
        WhisperDecodingOptions auto = WhisperDecodingOptions.defaults().withLanguage("auto");

        String fixed = transcriber.transcribe(englishAudio,
                WhisperDecodingOptions.defaults().withLanguage("en"), ignored -> { }).getText();
        String detectedEnglish = transcriber.transcribe(englishAudio, auto, ignored -> { }).getText();
        String detectedGerman = transcriber.transcribe(germanAudio, auto, ignored -> { }).getText();
        System.out.println("WHISPER-REGRESSION language | en: " + fixed + " | auto en: " + detectedEnglish
                + " | auto de: " + detectedGerman);

        assertTrue(wordErrorRate(english, fixed) <= MAXIMUM_WORD_ERROR_RATE, "English: " + fixed);
        assertTrue(wordErrorRate(english, detectedEnglish) <= MAXIMUM_WORD_ERROR_RATE, "auto en: " + detectedEnglish);
        assertTrue(wordErrorRate(SENTENCE, detectedGerman) <= MAXIMUM_WORD_ERROR_RATE, "auto de: " + detectedGerman);
    }

    private static int wordCount(String text) {
        return text.isBlank() ? 0 : text.trim().split("\s+").length;
    }

    private static float[] scaled(float[] samples, float gain) {
        float[] result = new float[samples.length];
        for (int i = 0; i < samples.length; i++) {
            result[i] = samples[i] * gain;
        }
        return result;
    }

    private static void assertQuality(String name, String reference, String text) {
        if (reference.isEmpty()) {
            assertEquals("", text, name + ": silence must not produce text");
            return;
        }
        assertTrue(wordErrorRate(reference, text) <= MAXIMUM_WORD_ERROR_RATE,
                name + ": too many recognition errors: " + text);
        assertTrue(longestRepetition(text) <= 2, name + ": repetition loop: " + text);
    }

    private static void report(String path, String name, double audioSeconds, long startedNanos,
                               String reference, String text) {
        double seconds = (System.nanoTime() - startedNanos) / 1e9;
        System.out.printf("WHISPER-REGRESSION %-8s | %-32s | audio %5.1f s | cpu %5.1f s | RTF %4.2f | "
                        + "WER %4.2f | repetition %d | %s%n", path, name, audioSeconds, seconds,
                seconds / audioSeconds, wordErrorRate(reference, text), longestRepetition(text), text);
    }

    /** Add -34 dBFS white noise on top of the regular microphone simulation. */
    private static float[] noisy(float[]... parts) {
        float[] audio = microphone(parts);
        Random random = new Random(7);
        for (int i = 0; i < audio.length; i++) {
            audio[i] += (float) (random.nextGaussian() * 0.02);
        }
        return audio;
    }
}
