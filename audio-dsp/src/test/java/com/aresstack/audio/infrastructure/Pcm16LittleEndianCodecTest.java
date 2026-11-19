package com.aresstack.audio.infrastructure;

import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class Pcm16LittleEndianCodecTest {

    @Test
    public void decodesKnownLittleEndianPattern() {
        // 0x0201 = 513, 0xFFFF = -1, 0x8000 = Short.MIN_VALUE, 0x7FFF = Short.MAX_VALUE
        byte[] bytes = {0x01, 0x02, (byte) 0xFF, (byte) 0xFF, 0x00, (byte) 0x80, (byte) 0xFF, 0x7F};
        short[] samples = new short[4];
        int count = Pcm16LittleEndianCodec.decode(bytes, bytes.length, samples);

        assertEquals(4, count);
        assertEquals(513, samples[0]);
        assertEquals(-1, samples[1]);
        assertEquals(Short.MIN_VALUE, samples[2]);
        assertEquals(Short.MAX_VALUE, samples[3]);
    }

    @Test
    public void roundTripProducesIdenticalBytes() {
        Random random = new Random(7);
        byte[] original = new byte[2048];
        random.nextBytes(original);

        short[] samples = new short[original.length / 2];
        int sampleCount = Pcm16LittleEndianCodec.decode(original, original.length, samples);
        byte[] roundTripped = new byte[original.length];
        int byteCount = Pcm16LittleEndianCodec.encode(samples, sampleCount, roundTripped);

        assertEquals(original.length, byteCount);
        assertArrayEquals(original, roundTripped);
    }
}
