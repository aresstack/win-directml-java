package com.aresstack.windirectml.inference.decoderonly;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DecoderOnlyStopTokenPolicyTest {

    @Test
    void fromTokenIdsMatchesConfiguredStopTokens() {
        DecoderOnlyStopTokenPolicy policy = DecoderOnlyStopTokenPolicy.fromTokenIds(Arrays.asList(1, 2, 3));

        assertTrue(policy.shouldStop(1));
        assertTrue(policy.shouldStop(3));
        assertFalse(policy.shouldStop(4));
    }
}
