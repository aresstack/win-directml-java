package com.aresstack.audio.pipeline;

import com.aresstack.audio.profile.AudioBlockType;

import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertFalse;

/** Verify that every registry-driven block and setting has user-facing help text. */
public class AudioDspHelpTest {

    @Test
    public void everyBlockAndParameterHasHelpText() {
        AudioBlockRegistry registry = AudioBlockRegistry.getInstance();
        for (AudioBlockType type : AudioBlockType.values()) {
            assertPresent("block " + type, AudioDspHelp.blockDescription(type));
            List<AudioParameterDescriptor> parameters = registry.descriptor(type).getParameters();
            for (int i = 0; i < parameters.size(); i++) {
                AudioParameterDescriptor parameter = parameters.get(i);
                assertPresent(type + "." + parameter.getKey(),
                        AudioDspHelp.parameterDescription(type, parameter.getKey()));
            }
        }
    }

    private static void assertPresent(String subject, String text) {
        assertFalse("missing help for " + subject, text == null || text.trim().isEmpty());
    }
}
