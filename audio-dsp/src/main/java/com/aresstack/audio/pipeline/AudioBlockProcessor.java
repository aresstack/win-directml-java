package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.profile.AudioBlockDefinition;

/**
 * Process a whole PCM buffer for one block. Buffer-level processors may change the audio format (a
 * resampler changes the rate, a channel mixer changes the channel count) by returning a new
 * {@link AudioBuffer}; a filter that keeps the format may process in place and return the same buffer.
 *
 * <p>A fresh processor is created for every pipeline run (see {@link AudioBlockDescriptor#createProcessor()}),
 * so stateful filters never leak state between recordings or threads. Adaptive/spectral blocks that need
 * frame-based work wrap a {@link StatefulAudioBlockProcessor} via {@link OverlapAddFramer} instead of
 * implementing this directly.</p>
 */
public interface AudioBlockProcessor {

    AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context);
}
