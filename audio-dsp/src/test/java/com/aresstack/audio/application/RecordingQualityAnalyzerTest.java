package com.aresstack.audio.application;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** Quality verdicts and their upload-blocking semantics. */
public class RecordingQualityAnalyzerTest {

    private final RecordingQualityAnalyzer analyzer = RecordingQualityAnalyzer.withDefaults();

    @Test
    public void tooShortBlocksUpload() {
        RecordingQuality quality = analyzer.analyze(120L, 2000.0d, 8000, 0L, 16000L, 0L);
        assertEquals(RecordingQuality.TOO_SHORT, quality);
        assertTrue(quality.blocksUpload());
    }

    @Test
    public void silenceIsNoSignalAndBlocksUpload() {
        RecordingQuality quality = analyzer.analyze(2000L, 5.0d, 40, 0L, 32000L, 0L);
        assertEquals(RecordingQuality.NO_SIGNAL, quality);
        assertTrue(quality.blocksUpload());
    }

    @Test
    public void clippingIsWarningNotBlocking() {
        // 3% of samples clipped, well above the 0.5% threshold.
        RecordingQuality quality = analyzer.analyze(2000L, 6000.0d, 32767, 960L, 32000L, 0L);
        assertEquals(RecordingQuality.CLIPPED, quality);
        assertFalse(quality.blocksUpload());
    }

    @Test
    public void droppedFramesReported() {
        RecordingQuality quality = analyzer.analyze(2000L, 3000.0d, 20000, 0L, 32000L, 5L);
        assertEquals(RecordingQuality.DROPPED_FRAMES, quality);
        assertFalse(quality.blocksUpload());
    }

    @Test
    public void cleanRecordingIsValid() {
        RecordingQuality quality = analyzer.analyze(2000L, 3000.0d, 20000, 0L, 32000L, 0L);
        assertEquals(RecordingQuality.VALID, quality);
        assertFalse(quality.blocksUpload());
    }
}
