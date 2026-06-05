package com.aresstack.windirectml.inference.t5;

import org.junit.jupiter.api.Test;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class T5StopTokenPolicyTest {

    @Test
    void stopsAtEosToken() {
        T5StopTokenPolicy policy = T5StopTokenPolicy.stopAtEos(2);

        assertTrue(policy.shouldStop(2));
        assertFalse(policy.shouldStop(3));
    }

    @Test
    void stopsAtAnyConfiguredToken() {
        T5StopTokenPolicy policy = T5StopTokenPolicy.stopAtAny(Set.of(2, 7));

        assertTrue(policy.shouldStop(2));
        assertTrue(policy.shouldStop(7));
        assertFalse(policy.shouldStop(8));
    }

    @Test
    void neverStopDoesNotStop() {
        T5StopTokenPolicy policy = T5StopTokenPolicy.neverStop();

        assertFalse(policy.shouldStop(2));
    }
}
