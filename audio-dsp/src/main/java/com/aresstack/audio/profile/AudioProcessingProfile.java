package com.aresstack.audio.profile;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Describe one ordered and reusable audio-processing pipeline. */
public final class AudioProcessingProfile {

    private final String id;
    private final String name;
    private final boolean builtIn;
    private final List<AudioBlockDefinition> blocks;

    public AudioProcessingProfile(String id, String name, boolean builtIn,
                                  List<AudioBlockDefinition> blocks) {
        if (id == null || id.trim().isEmpty()) {
            throw new IllegalArgumentException("Profile id must not be empty.");
        }
        if (name == null || name.trim().isEmpty()) {
            throw new IllegalArgumentException("Profile name must not be empty.");
        }
        if (blocks == null) {
            throw new IllegalArgumentException("Profile blocks must not be null.");
        }
        this.id = id.trim();
        this.name = name.trim();
        this.builtIn = builtIn;
        this.blocks = Collections.unmodifiableList(new ArrayList<AudioBlockDefinition>(blocks));
    }

    public String getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public boolean isBuiltIn() {
        return builtIn;
    }

    public List<AudioBlockDefinition> getBlocks() {
        return blocks;
    }

    public AudioProcessingProfile withBlocks(List<AudioBlockDefinition> value) {
        return new AudioProcessingProfile(id, name, builtIn, value);
    }

    public AudioProcessingProfile asUserProfile(String newId, String newName) {
        return new AudioProcessingProfile(newId, newName, false, blocks);
    }

    @Override
    public String toString() {
        return name;
    }
}
