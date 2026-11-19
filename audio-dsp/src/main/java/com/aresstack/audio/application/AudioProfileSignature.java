package com.aresstack.audio.application;

import com.aresstack.audio.profile.AudioBlockDefinition;
import com.aresstack.audio.profile.AudioProcessingProfile;

import java.util.Map;
import java.util.TreeMap;

/**
 * Build a stable signature of a profile's PIPELINE (block types, order, enabled flags and parameters) so a
 * preview result can be marked outdated when the editor pipeline changes. The signature deliberately
 * ignores the profile id and name — an unsaved edit changes the blocks, not the identity.
 */
public final class AudioProfileSignature {

    private AudioProfileSignature() {
    }

    public static String of(AudioProcessingProfile profile) {
        if (profile == null) {
            return "";
        }
        StringBuilder builder = new StringBuilder();
        for (AudioBlockDefinition block : profile.getBlocks()) {
            builder.append(block.getType().name()).append('|').append(block.isEnabled()).append('|');
            Map<String, String> sorted = new TreeMap<String, String>(block.getParameters());
            for (Map.Entry<String, String> entry : sorted.entrySet()) {
                builder.append(entry.getKey()).append('=').append(entry.getValue()).append(',');
            }
            builder.append(';');
        }
        return builder.toString();
    }
}
