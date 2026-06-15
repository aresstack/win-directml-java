package com.aresstack.windirectml.inference.gemma;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** GEMMA-WARP-11: runtime-mode selection. External is the default; native-warp only when requested. */
class Gemma3RuntimeModeTest {

    @Test
    void defaultsToExternalWhenUnset() {
        String prev = System.getProperty(Gemma3RuntimeMode.PROPERTY);
        System.clearProperty(Gemma3RuntimeMode.PROPERTY);
        try {
            assertEquals(Gemma3RuntimeMode.EXTERNAL, Gemma3RuntimeMode.fromSystemProperty());
        } finally {
            if (prev != null) {
                System.setProperty(Gemma3RuntimeMode.PROPERTY, prev);
            }
        }
    }

    @Test
    void nativeWarpFlagSelectsNativeWarp() {
        String prev = System.getProperty(Gemma3RuntimeMode.PROPERTY);
        System.setProperty(Gemma3RuntimeMode.PROPERTY, "native-warp");
        try {
            assertEquals(Gemma3RuntimeMode.NATIVE_WARP, Gemma3RuntimeMode.fromSystemProperty());
        } finally {
            if (prev == null) {
                System.clearProperty(Gemma3RuntimeMode.PROPERTY);
            } else {
                System.setProperty(Gemma3RuntimeMode.PROPERTY, prev);
            }
        }
    }

    @Test
    void valueParsing() {
        assertEquals(Gemma3RuntimeMode.EXTERNAL, Gemma3RuntimeMode.fromValue(null));
        assertEquals(Gemma3RuntimeMode.EXTERNAL, Gemma3RuntimeMode.fromValue(""));
        assertEquals(Gemma3RuntimeMode.EXTERNAL, Gemma3RuntimeMode.fromValue("external"));
        assertEquals(Gemma3RuntimeMode.EXTERNAL, Gemma3RuntimeMode.fromValue("something-else"));
        assertEquals(Gemma3RuntimeMode.NATIVE_WARP, Gemma3RuntimeMode.fromValue("native-warp"));
        assertEquals(Gemma3RuntimeMode.NATIVE_WARP, Gemma3RuntimeMode.fromValue("NATIVE-WARP"));
    }
}
