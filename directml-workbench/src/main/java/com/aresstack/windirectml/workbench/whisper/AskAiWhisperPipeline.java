package com.aresstack.windirectml.workbench.whisper;

import com.aresstack.audio.profile.AudioProcessingProfile;
import com.aresstack.windirectml.inference.whisper.TranscriptionResult;
import com.aresstack.windirectml.inference.whisper.TranscriptionSegment;
import com.aresstack.windirectml.inference.whisper.WhisperAudioSource;
import com.aresstack.windirectml.inference.whisper.WhisperTranscriber;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Coordinate the Ask AI VAD, configurable PCM processing profile and Whisper.
 * Skip gaps rather than concatenating them, preserving original timestamp positions.
 */
final class AskAiWhisperPipeline {
    private static final int SAMPLE_RATE = 16000;

    private final SpeechRegionDetector regionDetector;
    private final AudioProcessingProfile profile;

    AskAiWhisperPipeline() {
        this(new SpeechRegionDetector(), AskAiProcessedSpeechSource.speechProfile());
    }

    AskAiWhisperPipeline(SpeechRegionDetector regionDetector, AudioProcessingProfile profile) {
        this.regionDetector = Objects.requireNonNull(regionDetector, "regionDetector");
        this.profile = Objects.requireNonNull(profile, "profile");
    }

    TranscriptionResult transcribeWithTimestamps(WhisperAudioSource source,
                                                 WhisperTranscriber transcriber,
                                                 int maximumTokens,
                                                 Consumer<TranscriptionResult> progress)
            throws IOException {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(transcriber, "transcriber");
        Objects.requireNonNull(progress, "progress");
        List<TranscriptionResult> recognized = new ArrayList<>();
        for (SpeechInterval interval : regionDetector.detect(source)) {
            WhisperAudioSource prepared = new AskAiProcessedSpeechSource(source, interval, profile);
            long offsetMillis = toMillis(interval.startSample());
            TranscriptionResult region = transcriber.transcribeWithTimestamps(prepared,
                    maximumTokens, part -> progress.accept(shift(part, offsetMillis)));
            recognized.add(shift(region, offsetMillis));
        }
        return combine(recognized);
    }

    String transcribePlain(WhisperAudioSource source, WhisperTranscriber transcriber,
                           int maximumTokens) throws IOException {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(transcriber, "transcriber");
        StringBuilder text = new StringBuilder();
        for (SpeechInterval interval : regionDetector.detect(source)) {
            WhisperAudioSource prepared = new AskAiProcessedSpeechSource(source, interval, profile);
            append(text, transcriber.transcribeShortRecording(prepared, maximumTokens));
        }
        return text.toString();
    }

    static TranscriptionResult shift(TranscriptionResult result, long offsetMillis) {
        List<TranscriptionSegment> shifted = new ArrayList<>();
        for (TranscriptionSegment segment : result.getSegments()) {
            shifted.add(new TranscriptionSegment(segment.getStartMillis() + offsetMillis,
                    segment.getEndMillis() + offsetMillis, segment.getText()));
        }
        return new TranscriptionResult(result.getText(), result.getUnalignedText(), shifted);
    }

    private static TranscriptionResult combine(List<TranscriptionResult> regions) {
        StringBuilder text = new StringBuilder();
        StringBuilder unaligned = new StringBuilder();
        List<TranscriptionSegment> segments = new ArrayList<>();
        for (TranscriptionResult region : regions) {
            append(text, region.getText());
            append(unaligned, region.getUnalignedText());
            segments.addAll(region.getSegments());
        }
        return new TranscriptionResult(text.toString(), unaligned.toString(), segments);
    }

    private static void append(StringBuilder result, String text) {
        if (text == null || text.isBlank()) {
            return;
        }
        if (result.length() > 0) {
            result.append(' ');
        }
        result.append(text.strip());
    }

    private static long toMillis(long samplePosition) {
        return samplePosition * 1000L / SAMPLE_RATE;
    }
}
