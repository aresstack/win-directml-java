package com.aresstack.audio.pipeline;

import com.aresstack.audio.profile.AudioBlockDefinition;
import com.aresstack.audio.profile.AudioBlockType;

import java.util.List;
import java.util.Map;

/**
 * The single description of one block type, used everywhere a block is handled: to create its processor,
 * its default definition and parameters, to render inspector fields and the canvas summary, to validate,
 * and to group it in the editor. Registering a new block here replaces the former parallel per-type
 * switches in the processor, inspector and canvas.
 */
public interface AudioBlockDescriptor {

    AudioBlockType getType();

    String getTypeId();

    String getDisplayName();

    AudioBlockCategory getCategory();

    List<AudioParameterDescriptor> getParameters();

    /** @return the default parameter map (key -> default value) derived from {@link #getParameters()}. */
    Map<String, String> defaultParameters();

    AudioBlockDefinition createDefaultDefinition(String id);

    /** @return a FRESH processor for one pipeline run (never shared between recordings or threads). */
    AudioBlockProcessor createProcessor();

    AudioBlockCapabilities getCapabilities();

    /** @return a short one-line parameter summary for the canvas block. */
    String summarize(AudioBlockDefinition block);
}
