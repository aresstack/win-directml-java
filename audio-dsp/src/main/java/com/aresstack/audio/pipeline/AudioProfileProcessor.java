package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.profile.AudioBlockDefinition;
import com.aresstack.audio.profile.AudioProcessingProfile;

import java.util.List;

/**
 * Execute a stored profile over a complete PCM buffer, block by block, resolving each block's processor
 * through the {@link AudioBlockRegistry}. There is no per-type switch here: adding a block type means
 * registering a descriptor. Enabled blocks run; disabled blocks are skipped so a bypassed block never
 * changes the signal. A fresh processor is created per block per run, so stateful filters never share
 * state between recordings or threads.
 */
public final class AudioProfileProcessor {

    private final AudioBlockRegistry registry;

    public AudioProfileProcessor() {
        this(AudioBlockRegistry.getInstance());
    }

    public AudioProfileProcessor(AudioBlockRegistry registry) {
        if (registry == null) {
            throw new IllegalArgumentException("Registry must not be null.");
        }
        this.registry = registry;
    }

    public AudioBuffer process(AudioBuffer input, AudioProcessingProfile profile) {
        if (input == null) {
            throw new IllegalArgumentException("Input buffer must not be null.");
        }
        if (profile == null) {
            throw new IllegalArgumentException("Profile must not be null.");
        }
        AudioProcessingContext context = new AudioProcessingContext();
        AudioBuffer current = new AudioBuffer(copy(input.getSamples()), input.getFormat());
        List<AudioBlockDefinition> blocks = profile.getBlocks();
        for (int i = 0; i < blocks.size(); i++) {
            AudioBlockDefinition block = blocks.get(i);
            if (!block.isEnabled()) {
                continue;
            }
            AudioBlockProcessor processor = registry.createProcessor(block.getType());
            current = processor.process(current, block, context);
        }
        return current;
    }

    private static short[] copy(short[] input) {
        short[] copy = new short[input.length];
        System.arraycopy(input, 0, copy, 0, input.length);
        return copy;
    }
}
