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
    void advancesFullWindowWhenTheOnlySegmentSpansTheWholeWindow() {
        WhisperTimestampWindowStitcher stitcher = new WhisperTimestampWindowStitcher(60 * 16000);
        stitcher.accept(result(segment(0, 30, "Single segment.")));

        assertTrue(stitcher.hasNextWindow());
        assertEquals(30 * 16000, stitcher.nextStartSample());
        assertEquals(1, stitchedSegments(stitcher).size());
    }

    @Test
    void decodesALongBoundarySegmentAgainFromItsStart() {
        WhisperTimestampWindowStitcher stitcher = new WhisperTimestampWindowStitcher(60 * 16000);
        TranscriptionResult committed = stitcher.accept(result(segment(0, 3, "First."),
                segment(3, 30, "Cut by the window.")));

        assertEquals(3 * 16000, stitcher.nextStartSample());
        assertEquals("First.", committed.getText());
    }

    @Test
    void resumesAfterASingleWordInsteadOfSkippingTheRestOfTheWindow() {
        WhisperTimestampWindowStitcher stitcher = new WhisperTimestampWindowStitcher(60 * 16000);
        TranscriptionResult committed = stitcher.accept(result(segment(1, 2, "Hallo.")));

        assertEquals("Hallo.", committed.getText());
        assertEquals(2 * 16000, stitcher.nextStartSample(),
                "Audio after the word must be decoded again, not skipped");
    }

    @Test
    void resumesAtTheLastClosedSegmentWhenTextRunsIntoTheBoundary() {
        WhisperTimestampWindowStitcher stitcher = new WhisperTimestampWindowStitcher(60 * 16000);
        TranscriptionResult withOpenText = new TranscriptionResult("First. Second. Open",
                "Open", Arrays.asList(segment(0, 10, "First."), segment(10, 22, "Second.")));

        TranscriptionResult committed = stitcher.accept(withOpenText, true);

        assertEquals("First. Second.", committed.getText());
        assertEquals(22 * 16000, stitcher.nextStartSample());
    }

    @Test
    void keepsOpenTextOfTheFinalWindow() {
        WhisperTimestampWindowStitcher stitcher = new WhisperTimestampWindowStitcher(20 * 16000);
        TranscriptionResult withOpenText = new TranscriptionResult("First. Open end",
                "Open end", Arrays.asList(segment(0, 10, "First.")));

        assertEquals("First. Open end", stitcher.accept(withOpenText, true).getText());
        assertFalse(stitcher.hasNextWindow());
    }

    @Test
    void keepsUntimedTextRunningIntoTheBoundaryAndContinuesAfterTheWindow() {
        WhisperTimestampWindowStitcher stitcher = new WhisperTimestampWindowStitcher(45 * 16000);
        TranscriptionResult untimed = new TranscriptionResult("Ein langer Satz ohne Ende bis zur Gren",
                "Ein langer Satz ohne Ende bis zur Gren", Collections.emptyList());

        TranscriptionResult committed = stitcher.accept(untimed, true);

        assertEquals("Ein langer Satz ohne Ende bis zur Gren", committed.getText(),
                "Without any timestamp the text cannot be re-anchored and must not be dropped");
        assertEquals(30 * 16000, stitcher.nextStartSample());
        assertTrue(stitcher.hasNextWindow());
    }

    @Test
    void guaranteesProgressWhenTheModelStopsWithinTheFirstSecond() {
        WhisperTimestampWindowStitcher stitcher = new WhisperTimestampWindowStitcher(60 * 16000);
        stitcher.accept(new TranscriptionResult("Ja", "",
                Arrays.asList(new TranscriptionSegment(0, 400, "Ja"))));

        assertEquals(30 * 16000, stitcher.nextStartSample());
    }

    @Test
    void advancesFullWindowWithoutUsableTimestamps() {
        WhisperTimestampWindowStitcher stitcher = new WhisperTimestampWindowStitcher(60 * 16000);
        stitcher.accept(new TranscriptionResult("Unaligned speech", "Unaligned speech",
                Collections.emptyList()));

        assertEquals(30 * 16000, stitcher.nextStartSample());
        assertEquals("Unaligned speech", stitcher.acceptedWindows().get(0).getText());
    }

    @Test
    void treatsEveryWindowOfAnOpenRecordingAsNonFinalUntilItIsCompleted() {
        WhisperTimestampWindowStitcher stitcher = WhisperTimestampWindowStitcher.open();
        assertEquals(30 * 16000, stitcher.nextWindowSamples());

        TranscriptionResult committed = stitcher.accept(result(segment(0, 12, "First."), segment(12, 30, "Cut.")));
        assertEquals("First.", committed.getText(), "A boundary segment of an open recording is withheld");
        assertEquals(12 * 16000, stitcher.nextStartSample());

        stitcher.complete(20 * 16000);
        assertEquals(8 * 16000, stitcher.nextWindowSamples());
        TranscriptionResult last = stitcher.accept(result(segment(12, 19, "Cut and finished."),
                segment(19, 20, "End.")));
        assertEquals("Cut and finished. End.", last.getText());
        assertFalse(stitcher.hasNextWindow());
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
