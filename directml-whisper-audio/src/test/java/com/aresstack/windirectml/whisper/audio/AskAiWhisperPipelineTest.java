package com.aresstack.windirectml.whisper.audio;

import com.aresstack.windirectml.inference.whisper.TranscriptionResult;
import com.aresstack.windirectml.inference.whisper.TranscriptionSegment;
import com.aresstack.windirectml.inference.whisper.WhisperAudioSource;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AskAiWhisperPipelineTest {
    private static final int SAMPLE_RATE = 16000;

    @Test
    void skipsLongPausesWithoutMovingOriginalSpeechPositions() throws Exception {
        float[] audio = new float[SAMPLE_RATE * 5];
        fillTone(audio, SAMPLE_RATE, SAMPLE_RATE / 2);
        fillTone(audio, SAMPLE_RATE * 3, SAMPLE_RATE / 2);

        List<SpeechInterval> intervals = new SpeechRegionDetector().detect(source(audio));

        assertEquals(2, intervals.size());
        assertTrue(intervals.get(0).startSample() <= SAMPLE_RATE);
        assertTrue(intervals.get(0).endSample() >= SAMPLE_RATE + SAMPLE_RATE / 2);
        assertTrue(intervals.get(1).startSample() <= SAMPLE_RATE * 3L);
        assertTrue(intervals.get(1).endSample() >= SAMPLE_RATE * 3L + SAMPLE_RATE / 2);
        assertTrue(intervals.get(0).endSample() < intervals.get(1).startSample());
    }

    @Test
    void preservesASingleWordAtTheBeginningOfARecording() throws Exception {
        float[] audio = new float[SAMPLE_RATE * 2];
        fillTone(audio, 0, SAMPLE_RATE / 2);

        List<SpeechInterval> intervals = new SpeechRegionDetector().detect(source(audio));

        assertFalse(intervals.isEmpty());
        assertEquals(0, intervals.get(0).startSample());
        assertTrue(intervals.get(0).endSample() >= SAMPLE_RATE / 2);
    }

    @Test
    void silenceDoesNotProduceAnArtificialSpeechInterval() throws Exception {
        List<SpeechInterval> intervals = new SpeechRegionDetector().detect(source(new float[SAMPLE_RATE * 3]));
        assertTrue(intervals.isEmpty());
    }

    @Test
    void audioProfilePreservesSampleCountAndOriginalPosition() throws Exception {
        float[] audio = new float[SAMPLE_RATE * 3];
        fillTone(audio, SAMPLE_RATE, SAMPLE_RATE);
        SpeechInterval interval = new SpeechInterval(SAMPLE_RATE, SAMPLE_RATE * 2L);

        WhisperAudioSource filtered = new AskAiProcessedSpeechSource(
                source(audio), interval, AskAiProcessedSpeechSource.speechProfile(true));
        float[] processed = filtered.readSamples(0, SAMPLE_RATE);

        assertEquals(SAMPLE_RATE, processed.length);
        assertEquals(SAMPLE_RATE, filtered.sampleCount());
        assertFalse(isSilent(processed));
    }

    @Test
    void reducesLowFrequencyHumWithoutRemovingSpeechFrequencies() throws Exception {
        float[] hum = sineWave(45, SAMPLE_RATE, 0.2f);
        float[] speech = sineWave(440, SAMPLE_RATE, 0.2f);

        float[] filteredHum = filtered(hum);
        float[] filteredSpeech = filtered(speech);

        double humRatio = energy(filteredHum, SAMPLE_RATE / 5) / energy(hum, SAMPLE_RATE / 5);
        double speechRatio = energy(filteredSpeech, SAMPLE_RATE / 5) / energy(speech, SAMPLE_RATE / 5);
        assertTrue(humRatio < 0.65d, "Reduce the 45-Hz hum substantially");
        assertTrue(speechRatio > 0.75d, "Preserve speech-band frequencies");
    }

    @Test
    void windowedReadsEqualContinuousFilteringOfTheWholeRecording() throws Exception {
        float[] audio = new float[SAMPLE_RATE * 6];
        java.util.Random random = new java.util.Random(3);
        for (int i = 0; i < audio.length; i++) {
            audio[i] = 0.2f * (float) Math.sin(2 * Math.PI * 30 * i / SAMPLE_RATE)
                    + 0.1f * (float) Math.sin(2 * Math.PI * 700 * i / SAMPLE_RATE)
                    + 0.02f * (float) random.nextGaussian() + 0.05f;
        }
        SpeechInterval interval = new SpeechInterval(SAMPLE_RATE, audio.length);
        WhisperAudioSource filtered = new AskAiProcessedSpeechSource(
                source(audio), interval, AskAiWhisperPipeline.defaultSpeechProfile());

        float[] whole = filtered.readSamples(0, (int) filtered.sampleCount());
        for (int start = 0; start < whole.length; start += 12345) {
            int count = Math.min(12345, whole.length - start);
            float[] window = filtered.readSamples(start, count);
            for (int i = 0; i < count; i++) {
                assertEquals(whole[start + i], window[i], 2.0f / 32768,
                        "Filter state restarted at window offset " + start);
            }
        }
    }

    @Test
    void defaultProfileOnlyRemovesDcOffsetAndRumble() {
        List<com.aresstack.audio.profile.AudioBlockType> blocks = new java.util.ArrayList<>();
        AskAiWhisperPipeline.defaultSpeechProfile().getBlocks().forEach(block -> blocks.add(block.getType()));

        assertEquals(Arrays.asList(com.aresstack.audio.profile.AudioBlockType.DC_OFFSET_REMOVAL,
                com.aresstack.audio.profile.AudioBlockType.HIGH_PASS), blocks);
        assertEquals(1, AskAiWhisperPipeline.withoutHighPassProfile().getBlocks().size());
    }

    @Test
    void defaultProfileLeavesSpeechBandLevelsUnchanged() throws Exception {
        float[] quiet = sineWave(1000, SAMPLE_RATE, 0.01f);
        float[] loud = sineWave(1000, SAMPLE_RATE, 0.5f);

        double quietRatio = energy(filtered(quiet), SAMPLE_RATE / 5) / energy(quiet, SAMPLE_RATE / 5);
        double loudRatio = energy(filtered(loud), SAMPLE_RATE / 5) / energy(loud, SAMPLE_RATE / 5);
        assertEquals(1.0, quietRatio, 0.05, "No gate may attenuate quiet speech");
        assertEquals(1.0, loudRatio, 0.05, "No compressor or limiter may reduce loud speech");
    }

    @Test
    void shiftsTranscriptSegmentsToOriginalRecordingTime() {
        TranscriptionResult relative = new TranscriptionResult("nach oben", "",
                Arrays.asList(new TranscriptionSegment(100, 600, "nach oben")));

        TranscriptionResult shifted = AskAiWhisperPipeline.shift(relative, 7200);

        assertEquals("nach oben", shifted.getText());
        assertEquals(7300L, shifted.getSegments().get(0).getStartMillis());
        assertEquals(7800L, shifted.getSegments().get(0).getEndMillis());
    }

    private static float[] filtered(float[] samples) throws Exception {
        WhisperAudioSource source = new AskAiProcessedSpeechSource(source(samples),
                new SpeechInterval(0, samples.length), AskAiWhisperPipeline.defaultSpeechProfile());
        return source.readSamples(0, samples.length);
    }

    private static float[] sineWave(double frequencyHz, int length, float amplitude) {
        float[] samples = new float[length];
        for (int i = 0; i < length; i++) {
            samples[i] = amplitude * (float) Math.sin(2 * Math.PI * frequencyHz * i / SAMPLE_RATE);
        }
        return samples;
    }

    private static double energy(float[] samples, int skippedSamples) {
        double sum = 0;
        for (int i = skippedSamples; i < samples.length; i++) {
            sum += (double) samples[i] * samples[i];
        }
        return Math.sqrt(sum / Math.max(1, samples.length - skippedSamples));
    }

    private static boolean isSilent(float[] samples) {
        for (float sample : samples) {
            if (Math.abs(sample) > 0.001f) {
                return false;
            }
        }
        return true;
    }

    private static void fillTone(float[] samples, int start, int count) {
        for (int i = 0; i < count; i++) {
            samples[start + i] = 0.4f * (float) Math.sin(2 * Math.PI * 450 * i / SAMPLE_RATE);
        }
    }

    private static WhisperAudioSource source(float[] samples) {
        return new WhisperAudioSource() {
            @Override
            public long sampleCount() {
                return samples.length;
            }

            @Override
            public float[] readSamples(long startSample, int sampleCount) {
                return Arrays.copyOfRange(samples, Math.toIntExact(startSample),
                        Math.toIntExact(startSample) + sampleCount);
            }
        };
    }
}
