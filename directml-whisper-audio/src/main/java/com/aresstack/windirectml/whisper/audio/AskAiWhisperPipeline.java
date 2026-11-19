package com.aresstack.windirectml.whisper.audio;

import com.aresstack.audio.profile.AudioProcessingProfile;
import com.aresstack.windirectml.inference.whisper.TranscriptionResult;
import com.aresstack.windirectml.inference.whisper.TranscriptionSegment;
import com.aresstack.windirectml.inference.whisper.WhisperAudioSource;
import com.aresstack.windirectml.inference.whisper.WhisperDecodingOptions;
import com.aresstack.windirectml.inference.whisper.WhisperTranscriber;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Coordinate the Ask AI VAD, a timebase-preserving PCM profile and the guarded Whisper decoder.
 *
 * <p>The default profile removes only DC offset and rumble below 80 Hz. The VAD skips long
 * silence between speech regions; every region keeps its original recording positions and is
 * transcribed in windows of at most thirty seconds. Nothing is concatenated, so timestamps map
 * directly to the recording. Plain text is the timestamped transcript without timestamps.
 *
 * <p>When the VAD finds no speech at all in a non-empty recording, for example a whispered or
 * very quiet word, the whole recording is decoded instead. The decoder's no-speech detection
 * still returns an empty transcript for genuine silence.
 */
public final class AskAiWhisperPipeline {
    private static final int SAMPLE_RATE = 16000;

    private final SpeechRegionDetector regionDetector;
    private final AudioProcessingProfile profile;
    private final WhisperDecodingOptions options;

    /** Use the minimal speech profile and the default guarded decoding options. */
    public AskAiWhisperPipeline() {
        this(defaultSpeechProfile());
    }

    /** Use a custom timebase-preserving Ask AI audio processing profile. */
    public AskAiWhisperPipeline(AudioProcessingProfile profile) {
        this(profile, WhisperDecodingOptions.defaults());
    }

    public AskAiWhisperPipeline(AudioProcessingProfile profile, WhisperDecodingOptions options) {
        this(new SpeechRegionDetector(), profile, options);
    }

    AskAiWhisperPipeline(SpeechRegionDetector regionDetector, AudioProcessingProfile profile,
                         WhisperDecodingOptions options) {
        this.regionDetector = Objects.requireNonNull(regionDetector, "regionDetector");
        this.profile = Objects.requireNonNull(profile, "profile");
        this.options = Objects.requireNonNull(options, "options");
    }

    /** DC-offset removal and an 80-Hz Butterworth high-pass: the default speech preparation. */
    public static AudioProcessingProfile defaultSpeechProfile() {
        return AskAiProcessedSpeechSource.speechProfile(true);
    }

    /** DC-offset removal only, for sources that are already high-pass filtered. */
    public static AudioProcessingProfile withoutHighPassProfile() {
        return AskAiProcessedSpeechSource.speechProfile(false);
    }

    /**
     * The 0.3.1 cleanup chain with low-pass, noise gate, compressor and limiter. These blocks
     * alter the speech signal Whisper was trained on; use them only after measuring a benefit.
     */
    public static AudioProcessingProfile enhancedSpeechProfile() {
        return AskAiProcessedSpeechSource.enhancedSpeechProfile();
    }

    /** Transcribe detected speech regions and return segments with original recording times. */
    public TranscriptionResult transcribe(WhisperAudioSource source, WhisperTranscriber transcriber,
                                          Consumer<TranscriptionResult> progress) throws IOException {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(transcriber, "transcriber");
        Objects.requireNonNull(progress, "progress");
        List<TranscriptionResult> recognized = new ArrayList<>();
        for (SpeechInterval interval : speechRegions(source)) {
            WhisperAudioSource prepared = new AskAiProcessedSpeechSource(source, interval, profile);
            long offsetMillis = toMillis(interval.startSample());
            TranscriptionResult region = transcriber.transcribe(prepared, options,
                    part -> progress.accept(shift(part, offsetMillis)));
            recognized.add(shift(region, offsetMillis));
        }
        return combine(recognized);
    }

    /** Transcribe detected speech regions and return the transcript without timestamps. */
    public String transcribePlain(WhisperAudioSource source, WhisperTranscriber transcriber)
            throws IOException {
        return transcribe(source, transcriber, ignored -> { }).getText();
    }

    /**
     * @deprecated Since 0.3.2 the token limit belongs to {@link WhisperDecodingOptions};
     * use {@link #transcribe(WhisperAudioSource, WhisperTranscriber, Consumer)}.
     */
    @Deprecated
    public TranscriptionResult transcribeWithTimestamps(WhisperAudioSource source,
                                                        WhisperTranscriber transcriber,
                                                        int maximumTokens) throws IOException {
        return withTokenLimit(maximumTokens).transcribe(source, transcriber, ignored -> { });
    }

    /**
     * @deprecated Since 0.3.2 the token limit belongs to {@link WhisperDecodingOptions};
     * use {@link #transcribe(WhisperAudioSource, WhisperTranscriber, Consumer)}.
     */
    @Deprecated
    public TranscriptionResult transcribeWithTimestamps(WhisperAudioSource source,
                                                        WhisperTranscriber transcriber,
                                                        int maximumTokens,
                                                        Consumer<TranscriptionResult> progress)
            throws IOException {
        return withTokenLimit(maximumTokens).transcribe(source, transcriber, progress);
    }

    /**
     * @deprecated Since 0.3.2 plain text uses the guarded timestamp decoder;
     * use {@link #transcribePlain(WhisperAudioSource, WhisperTranscriber)}.
     */
    @Deprecated
    public String transcribePlain(WhisperAudioSource source, WhisperTranscriber transcriber,
                                  int maximumTokens) throws IOException {
        return withTokenLimit(maximumTokens).transcribePlain(source, transcriber);
    }

    private AskAiWhisperPipeline withTokenLimit(int maximumTokens) {
        return new AskAiWhisperPipeline(regionDetector, profile,
                options.withMaximumTokensPerWindow(maximumTokens));
    }

    /** Return the VAD regions, or the complete recording when the VAD detected nothing. */
    List<SpeechInterval> speechRegions(WhisperAudioSource source) throws IOException {
        List<SpeechInterval> regions = regionDetector.detect(source);
        if (regions.isEmpty() && source.sampleCount() > 0) {
            return Collections.singletonList(new SpeechInterval(0, source.sampleCount()));
        }
        return regions;
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
