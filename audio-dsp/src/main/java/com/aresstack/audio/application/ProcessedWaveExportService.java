package com.aresstack.audio.application;

import java.io.File;
import java.io.IOException;

/** Save a processed preview as a WAV file. */
public interface ProcessedWaveExportService {

    void export(ProcessedAudioPreview preview, File targetFile) throws IOException;
}
