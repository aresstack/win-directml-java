package com.aresstack.windirectml.inference.t5;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class T5RuntimeRequestTest {

    @Test
    void rejectsEmptyInputTokens() {
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> new T5RuntimeRequest(new int[0], 1, T5StopTokenPolicy.neverStop(), 0, 0.0f, 0));

        assertTrue(error.getMessage().contains("inputTokenIds"));
    }

    @Test
    void rejectsNonPositiveMaxNewTokens() {
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> new T5RuntimeRequest(new int[]{1}, 0, T5StopTokenPolicy.neverStop(), 0, 0.0f, 0));

        assertTrue(error.getMessage().contains("maxNewTokens"));
    }
    @Test
    void greedyTextSuppressesDecoderControlTokensAndRequiresOneGeneratedTokenBeforeStop() {
        T5SpecialTokens specialTokens = new T5SpecialTokens(0, 1, 0);

        T5RuntimeRequest request = T5RuntimeRequest.greedyText(new int[]{3, 4}, 8, specialTokens);

        assertTrue(request.suppressedTokenIds().contains(0));
        assertEquals(1, request.minimumTokensBeforeStop());
    }

}
