package com.aresstack.audio.application;

import com.aresstack.audio.infrastructure.WavFileAudioSink;

import java.io.File;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

/**
 * Write a processed preview to a PCM-16 WAV, safely: first to a temp file next to the target, then move it
 * over the target atomically where the platform allows it. A failed write never leaves a partial target.
 * Overwrite confirmation is the caller's (UI) concern; this service just writes the given target path.
 */
public final class DefaultProcessedWaveExportService implements ProcessedWaveExportService {

    public void export(ProcessedAudioPreview preview, File targetFile) throws IOException {
        if (preview == null) {
            throw new IllegalArgumentException("Preview must not be null.");
        }
        if (targetFile == null) {
            throw new IllegalArgumentException("Target file must not be null.");
        }
        File directory = targetFile.getParentFile();
        if (directory != null && !directory.isDirectory() && !directory.mkdirs()) {
            throw new IOException("Cannot create directory for the WAV export: " + directory);
        }
        File temp = File.createTempFile(targetFile.getName() + ".", ".wav.tmp",
                directory != null ? directory : targetFile.getAbsoluteFile().getParentFile());
        boolean written = false;
        try {
            WavFileAudioSink sink = new WavFileAudioSink(temp);
            sink.open(preview.getFormat());
            try {
                sink.write(preview.getSamples(), preview.getSamples().length);
            } finally {
                sink.close();
            }
            try {
                Files.move(temp.toPath(), targetFile.toPath(),
                        StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException notAtomic) {
                Files.move(temp.toPath(), targetFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
            }
            written = true;
        } finally {
            if (!written && temp.exists() && !temp.delete()) {
                temp.deleteOnExit();
            }
        }
    }
}
