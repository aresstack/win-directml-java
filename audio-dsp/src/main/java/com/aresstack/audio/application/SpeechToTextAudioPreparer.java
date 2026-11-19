package com.aresstack.audio.application;

import com.aresstack.audio.domain.AudioBuffer;

import java.io.File;
import java.io.IOException;

/**
 * Produce the transport file a speech-to-text model actually expects, as a final step <em>after</em> the
 * (optional) DSP pipeline. This is deliberately separate from DSP processing: the DSP pipeline stays
 * format-neutral (it keeps the source sample rate and channel count unless an explicit block changes
 * them), while this component guarantees the proven STT input format regardless of what the pipeline
 * produced.
 *
 * <p>The {@code targetWav} is supplied by the caller so it owns temp-file placement and cleanup.</p>
 */
public interface SpeechToTextAudioPreparer {

    /**
     * Write {@code source} (already DSP-processed, in any rate/channel layout) into {@code targetWav} in
     * the format the STT backend expects, and return {@code targetWav}.
     */
    File prepare(AudioBuffer source, File targetWav) throws IOException;
}
