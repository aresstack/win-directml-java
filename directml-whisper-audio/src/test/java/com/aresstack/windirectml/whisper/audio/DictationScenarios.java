package com.aresstack.windirectml.whisper.audio;

import com.aresstack.windirectml.inference.whisper.WhisperAudioSource;
import com.aresstack.windirectml.inference.whisper.audio.WhisperWavAudioLoader;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Build reproducible microphone-like dictation recordings from synthesized German speech.
 * Add deterministic low-level noise and mains hum so silence is never digitally perfect.
 */
final class DictationScenarios {
    static final int SAMPLE_RATE = 16000;
    static final Path MODEL = Path.of("..", "model", "whisper-tiny");
    private static final Path FIXTURES = Path.of("src", "test", "resources", "whisper");

    static final String WORD = "Hallo.";
    static final String SENTENCE = "Ich dachte mir, du kannst mal im Internet nach weiteren Unterlagen "
            + "suchen und mir detaillierte Zusammenfassungen erstellen.";
    static final String PAUSE_A = "Bitte fasse die wichtigsten Punkte der Besprechung zusammen.";
    static final String PAUSE_B = "Danach schicke die Zusammenfassung an das gesamte Team.";
    static final String LONG = "Die Wissensdatenbank enthält zahlreiche Dokumente aus verschiedenen "
            + "Abteilungen. Jede Seite beschreibt ein Verfahren, eine Richtlinie oder ein technisches "
            + "Konzept. Die Mitarbeiter suchen regelmäßig nach Antworten auf konkrete Fragen, zum Beispiel "
            + "zur Urlaubsplanung, zur Reisekostenabrechnung oder zur Einrichtung eines neuen Arbeitsplatzes. "
            + "Damit die Suche zuverlässig funktioniert, werden alle Inhalte zunächst heruntergeladen, "
            + "anschließend in kleinere Abschnitte zerlegt und schließlich in einem lokalen Index gespeichert. "
            + "Ein Sprachmodell kann diese Abschnitte danach zusammenfassen und mit Quellenangaben beantworten. "
            + "Wichtig ist dabei, dass keine vertraulichen Daten das Unternehmen verlassen. Deshalb läuft die "
            + "gesamte Verarbeitung auf dem eigenen Rechner, ohne Verbindung zu externen Diensten. Auch die "
            + "Spracheingabe wird vollständig lokal ausgewertet, damit niemand außerhalb des Unternehmens die "
            + "Aufnahmen hören kann.";

    private DictationScenarios() {
    }

    static boolean modelAvailable() {
        return Files.isRegularFile(MODEL.resolve("model.safetensors"))
                && Files.isRegularFile(MODEL.resolve("vocab.json"));
    }

    static List<Scenario> all() throws IOException {
        List<Scenario> scenarios = new ArrayList<>();
        scenarios.add(new Scenario("single word", WORD,
                microphone(silence(1.0), speech("de-word.wav"), silence(1.0))));
        scenarios.add(new Scenario("short sentence", SENTENCE,
                microphone(silence(0.5), speech("de-sentence.wav"), silence(0.5))));
        scenarios.add(new Scenario("sentences with pauses", PAUSE_A + " " + PAUSE_B + " " + PAUSE_A,
                microphone(silence(1.0), speech("de-pause-a.wav"), silence(6.0),
                        speech("de-pause-b.wav"), silence(2.5), speech("de-pause-a.wav"), silence(1.0))));
        scenarios.add(new Scenario("continuous speech over 60 s", LONG,
                microphone(silence(0.5), speech("de-long.wav"), silence(0.5))));
        scenarios.add(new Scenario("speech across the 30 s boundary", String.join(" ",
                SENTENCE, SENTENCE, SENTENCE, SENTENCE),
                microphone(silence(0.5), speech("de-sentence.wav"), silence(0.5), speech("de-sentence.wav"),
                        silence(0.4), speech("de-sentence.wav"), silence(0.4), speech("de-sentence.wav"))));
        scenarios.add(new Scenario("silence only", "", microphone(silence(10.0))));
        return scenarios;
    }

    static float[] speech(String file) throws IOException {
        return new WhisperWavAudioLoader().load(FIXTURES.resolve(file));
    }

    static float[] silence(double seconds) {
        return new float[(int) Math.round(seconds * SAMPLE_RATE)];
    }

    /** Concatenate parts, then add -50 dBFS white noise and a -40 dBFS 50-Hz hum. */
    static float[] microphone(float[]... parts) {
        int length = Arrays.stream(parts).mapToInt(part -> part.length).sum();
        float[] audio = new float[length];
        int offset = 0;
        for (float[] part : parts) {
            System.arraycopy(part, 0, audio, offset, part.length);
            offset += part.length;
        }
        Random random = new Random(42);
        for (int i = 0; i < audio.length; i++) {
            audio[i] += (float) (random.nextGaussian() * 0.003
                    + 0.01 * Math.sin(2 * Math.PI * 50 * i / SAMPLE_RATE));
        }
        return audio;
    }

    static WhisperAudioSource source(float[] samples) {
        return new WhisperAudioSource() {
            @Override
            public long sampleCount() {
                return samples.length;
            }

            @Override
            public float[] readSamples(long startSample, int sampleCount) {
                int start = Math.toIntExact(startSample);
                return Arrays.copyOfRange(samples, start, start + sampleCount);
            }
        };
    }

    /** Word error rate after lower-casing and removing punctuation. */
    static double wordErrorRate(String reference, String hypothesis) {
        List<String> expected = words(reference);
        List<String> actual = words(hypothesis);
        if (expected.isEmpty()) {
            return actual.isEmpty() ? 0 : 1;
        }
        int[] previous = new int[actual.size() + 1];
        int[] current = new int[actual.size() + 1];
        for (int j = 0; j <= actual.size(); j++) {
            previous[j] = j;
        }
        for (int i = 1; i <= expected.size(); i++) {
            current[0] = i;
            for (int j = 1; j <= actual.size(); j++) {
                int substitution = previous[j - 1] + (expected.get(i - 1).equals(actual.get(j - 1)) ? 0 : 1);
                current[j] = Math.min(substitution, Math.min(previous[j] + 1, current[j - 1] + 1));
            }
            int[] swap = previous;
            previous = current;
            current = swap;
        }
        return (double) previous[actual.size()] / expected.size();
    }

    /** Longest run of one immediately repeated word or phrase of up to eight words. */
    static int longestRepetition(String text) {
        List<String> words = words(text);
        int longest = words.isEmpty() ? 0 : 1;
        for (int length = 1; length <= 8; length++) {
            for (int start = 0; start + length <= words.size(); start++) {
                int repetitions = 1;
                while (start + (repetitions + 1) * length <= words.size()
                        && words.subList(start, start + length).equals(
                        words.subList(start + repetitions * length, start + (repetitions + 1) * length))) {
                    repetitions++;
                }
                longest = Math.max(longest, repetitions);
            }
        }
        return longest;
    }

    private static List<String> words(String text) {
        List<String> words = new ArrayList<>();
        for (String word : text.toLowerCase(Locale.ROOT).split("[^\\p{L}\\p{N}]+")) {
            if (!word.isEmpty()) {
                words.add(word);
            }
        }
        return words;
    }

    static final class Scenario {
        final String name;
        final String reference;
        final float[] audio;

        Scenario(String name, String reference, float[] audio) {
            this.name = name;
            this.reference = reference;
            this.audio = audio;
        }

        double seconds() {
            return audio.length / (double) SAMPLE_RATE;
        }
    }
}
