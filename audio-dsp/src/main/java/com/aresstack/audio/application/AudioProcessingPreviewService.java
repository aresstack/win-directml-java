package com.aresstack.audio.application;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.profile.AudioProcessingProfile;

/**
 * Process one test source through a pipeline snapshot for interactive preview. Implementations must use
 * the SAME productive DSP path as microphone/file transcription — there is no separate simplified preview
 * pipeline.
 */
public interface AudioProcessingPreviewService {

    /**
     * @param source  the decoded raw test audio (unchanged by this call)
     * @param profile an immutable pipeline snapshot to apply
     * @param sourceId an identifier of the source, carried into the result for provenance
     * @return the processed preview (fresh adaptive state per call)
     */
    ProcessedAudioPreview process(AudioBuffer source, AudioProcessingProfile profile, String sourceId);
}
