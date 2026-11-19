package com.aresstack.windirectml.workbench.whisper;

import com.aresstack.windirectml.inference.whisper.TranscriptionResult;
import com.aresstack.windirectml.inference.whisper.TranscriptionSegment;

/** Format Whisper's model-aligned segments for the microphone test window. */
final class WhisperTranscriptFormatter {
    private WhisperTranscriptFormatter() {
    }

    static String format(TranscriptionResult result) {
        StringBuilder output = new StringBuilder();
        for (TranscriptionSegment segment : result.getSegments()) {
            output.append('[').append(formatTime(segment.getStartMillis()))
                    .append(" – ").append(formatTime(segment.getEndMillis()))
                    .append("] ").append(segment.getText()).append('\n');
        }
        if (result.hasUnalignedText()) {
            output.append("\nOhne bestätigte Zeitzuordnung: ")
                    .append(result.getUnalignedText()).append('\n');
        }
        if (output.length() == 0) {
            if (result.getText().isBlank()) {
                return "(Keine Sprache erkannt)";
            }
            return result.getText() + "\n\n(Whisper hat keine vollständigen Zeitstempel-Paare geliefert.)";
        }
        return output.toString().strip();
    }

    private static String formatTime(long milliseconds) {
        long minutes = milliseconds / 60000;
        long seconds = (milliseconds / 1000) % 60;
        long centiseconds = (milliseconds % 1000) / 10;
        return String.format(java.util.Locale.ROOT, "%02d:%02d.%02d",
                minutes, seconds, centiseconds);
    }
}
