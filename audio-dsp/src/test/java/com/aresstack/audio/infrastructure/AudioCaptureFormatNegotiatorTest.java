package com.aresstack.audio.infrastructure;

import com.aresstack.audio.application.AudioCaptureException;
import com.aresstack.audio.domain.PcmAudioFormat;

import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/** Negotiation picks the first supported candidate; none supported fails clearly. */
public class AudioCaptureFormatNegotiatorTest {

    @Test
    public void picksFirstSupportedFormat() throws Exception {
        // Device only supports 44.1 kHz stereo (the 4th default candidate).
        AudioCaptureFormatNegotiator negotiator = new AudioCaptureFormatNegotiator(
                (deviceName, format) -> format.getSampleRateHz() == 44100 && format.getChannels() == 2);
        PcmAudioFormat chosen = negotiator.negotiate("USB Mic", AudioCaptureFormatNegotiator.defaultCandidates());
        assertEquals(44100, chosen.getSampleRateHz());
        assertEquals(2, chosen.getChannels());
    }

    @Test
    public void prefersEarlierCandidateWhenMultipleSupported() throws Exception {
        AudioCaptureFormatNegotiator negotiator = new AudioCaptureFormatNegotiator(
                (deviceName, format) -> true); // everything supported → first candidate wins
        List<PcmAudioFormat> candidates = AudioCaptureFormatNegotiator.defaultCandidates();
        PcmAudioFormat chosen = negotiator.negotiate(null, candidates);
        assertEquals(candidates.get(0).getSampleRateHz(), chosen.getSampleRateHz());
        assertEquals(candidates.get(0).getChannels(), chosen.getChannels());
    }

    @Test
    public void throwsWhenNothingSupported() {
        AudioCaptureFormatNegotiator negotiator = new AudioCaptureFormatNegotiator(
                (deviceName, format) -> false);
        try {
            negotiator.negotiate("Broken Mic", AudioCaptureFormatNegotiator.defaultCandidates());
            fail("expected AudioCaptureException");
        } catch (AudioCaptureException expected) {
            // message names the device
        }
    }
}
