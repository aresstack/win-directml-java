package com.aresstack.windirectml.inference.whisper;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WhisperTimestampWindowStitcherTest {

    @Test
    void resumesAtLastSegmentStartAndPreservesAbsoluteTimestamps() {
        WhisperTimestampWindowStitcher stitcher = new WhisperTimestampWindowStitcher(70 * 16000);

        stitcher.accept(result(
                segment(0, 12, "First."),
                segment(12, 25, "Second."),
                segment(25, 30, "Incomplete.")));
        assertEquals(25 * 16000, stitcher.nextStartSample());

        stitcher.accept(result(
                segment(25, 38, "Complete."),
                segment(38, 50, "Next."),
                segment(50, 55, "Incomplete again.")));
        assertEquals(50 * 16000, stitcher.nextStartSample());

        stitcher.accept(result(
                segment(50, 60, "Final first."),
                segment(60, 70, "Final second.")));

        assertFalse(stitcher.hasNextWindow());
        List<TranscriptionSegment> segments = stitchedSegments(stitcher);
        assertEquals(Arrays.asList(0L, 12000L, 25000L, 38000L, 50000L, 60000L),
                segments.stream().map(TranscriptionSegment::getStartMillis)
                        .collect(Collectors.toList()));
        assertEquals(Arrays.asList("First.", "Second.", "Complete.", "Next.",
                        "Final first.", "Final second."),
                segments.stream().map(TranscriptionSegment::getText)
                        .collect(Collectors.toList()));
    }

    @Test
    void publishesOnlyCommittedSegmentsBeforeTheFinalWindow() {
        WhisperTimestampWindowStitcher stitcher = new WhisperTimestampWindowStitcher(50 * 16000);

        TranscriptionResult first = stitcher.accept(result(
                segment(0, 12, "First."),
                segment(12, 25, "Second."),
                segment(25, 30, "Withheld.")));

        assertEquals(Arrays.asList("First.", "Second."),
                first.getSegments().stream().map(TranscriptionSegment::getText)
                        .collect(Collectors.toList()));
        assertEquals("First. Second.", first.getText());
        assertEquals(25 * 16000, stitcher.nextStartSample());

        TranscriptionResult last = stitcher.accept(result(
                segment(25, 38, "Recovered."),
                segment(38, 50, "Final.")));
        assertEquals(Arrays.asList("Recovered.", "Final."),
                last.getSegments().stream().map(TranscriptionSegment::getText)
                        .collect(Collectors.toList()));
        assertFalse(stitcher.hasNextWindow());
    }

    @Test
    void commitsAllSegmentsInFinalWindow() {
        WhisperTimestampWindowStitcher stitcher = new WhisperTimestampWindowStitcher(30 * 16000);
        stitcher.accept(result(segment(0, 25, "First."), segment(25, 30, "Last.")));

        assertFalse(stitcher.hasNextWindow());
        assertEquals(2, stitchedSegments(stitcher).size());
    }

    @Test
    void advancesFullWindowWhenThereAreTooFewSegments() {
        WhisperTimestampWindowStitcher stitcher = new WhisperTimestampWindowStitcher(60 * 16000);
        stitcher.accept(result(segment(0, 30, "Single segment.")));

        assertTrue(stitcher.hasNextWindow());
        assertEquals(30 * 16000, stitcher.nextStartSample());
        assertEquals(1, stitchedSegments(stitcher).size());
    }

    @Test
    void advancesFullWindowWhenLastSegmentWouldCauseMinimalProgress() {
        WhisperTimestampWindowStitcher stitcher = new WhisperTimestampWindowStitcher(60 * 16000);
        stitcher.accept(result(segment(0, 3, "First."),
                segment(3, 30, "Very long second.")));

        assertEquals(30 * 16000, stitcher.nextStartSample());
        assertEquals(2, stitchedSegments(stitcher).size());
    }

    @Test
    void advancesFullWindowWithoutUsableTimestamps() {
        WhisperTimestampWindowStitcher stitcher = new WhisperTimestampWindowStitcher(60 * 16000);
        stitcher.accept(new TranscriptionResult("Unaligned speech", "Unaligned speech",
                Collections.emptyList()));

        assertEquals(30 * 16000, stitcher.nextStartSample());
        assertEquals("Unaligned speech", stitcher.acceptedWindows().get(0).getText());
    }

    private static TranscriptionResult result(TranscriptionSegment... segments) {
        String text = Arrays.stream(segments).map(TranscriptionSegment::getText)
                .collect(Collectors.joining(" "));
        return new TranscriptionResult(text, "", Arrays.asList(segments));
    }

    private static TranscriptionSegment segment(int startSeconds, int endSeconds, String text) {
        return new TranscriptionSegment(startSeconds * 1000L, endSeconds * 1000L, text);
    }

    private static List<TranscriptionSegment> stitchedSegments(WhisperTimestampWindowStitcher stitcher) {
        return stitcher.acceptedWindows().stream().flatMap(result -> result.getSegments().stream())
                .collect(Collectors.toList());
    }
}
