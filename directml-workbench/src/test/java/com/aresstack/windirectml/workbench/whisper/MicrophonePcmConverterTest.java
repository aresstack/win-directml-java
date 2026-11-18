package com.aresstack.windirectml.workbench.whisper;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MicrophonePcmConverterTest {
    @Test
    void decodesSignedLittleEndianPcmCorrectly() {
        byte[] bytes = {0, 0, (byte) 0xff, 0x7f, 0, (byte) 0x80, 0, 0x40};
        float[] samples = MicrophonePcmConverter.convert(bytes, 16000);
        assertEquals(4, samples.length);
        assertEquals(0f, samples[0], 0f);
        assertEquals(32767f / 32768f, samples[1], 0.000001f);
        assertEquals(-1f, samples[2], 0f);
        assertEquals(0.5f, samples[3], 0f);
    }

    @Test
    void resamplesFortyEightKilohertzRecording() {
        byte[] bytes = new byte[48000 * 2];
        float[] samples = MicrophonePcmConverter.convert(bytes, 48000);
        assertEquals(16000, samples.length);
    }

    @Test
    void rejectsTruncatedFrames() {
        assertThrows(IllegalArgumentException.class,
                () -> MicrophonePcmConverter.convert(new byte[3], 16000));
    }
}
